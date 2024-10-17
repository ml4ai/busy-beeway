from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from transformers.training.jax_utils import pref_loss_fn


class PrefTransformerTrainer(object):

    def __init__(self, trans, rng_key1, rng_key2, pretrained_params=None, **kwargs):
        self.trans = trans

        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=kwargs.get("init_value", 0),
            peak_value=kwargs.get("peak_value", 1e-4),
            warmup_steps=kwargs.get("warmup_steps", 65),
            decay_steps=kwargs.get("decay_steps", 650),
            end_value=kwargs.get("end_value", 0),
        )
        tx = optimizer_class(scheduler_class)
        # Reconfigure for our data
        if pretrained_params is None:
            trans_params = self.trans.init(
                {"params": rng_key1, "dropout": rng_key2},
                jnp.zeros((10, 25, trans.state_dim)),
                jnp.zeros((10, 25, trans.action_dim)),
                jnp.ones((10, 25), dtype=jnp.int32),
            )
            self._train_state = TrainState.create(
                params=trans_params, tx=tx, apply_fn=self.trans.apply
            )
        else:
            self._train_state = TrainState.create(
                params=pretrained_params, tx=tx, apply_fn=self.trans.apply
            )

    def evaluation(self, batch, rng_key):
        return _eval_pref_step(self._train_state, batch, rng_key)

    def train(self, batch, rng_key):
        self._train_state, metrics = _train_pref_step(self._train_state, batch, rng_key)
        return metrics

    def get_params(self):
        return self._train_state.params


@jax.jit
def _eval_pref_step(state, batch, rng_key):
    loss, acc = pref_loss_fn(state.apply_fn, state.params, batch, False, rng_key)
    return dict(eval_loss=loss, eval_acc=acc)


@jax.jit
def _train_pref_step(state, batch, rng_key):
    grad_fn = jax.value_and_grad(pref_loss_fn, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state.apply_fn, state.params, batch, True, rng_key)
    new_train_state = state.apply_gradients(grads=grads)
    metrics = dict(training_loss=loss, training_acc=acc)
    return new_train_state, metrics


class MAMLPTTrainer(object):

    def __init__(self, trans, rng_key1, rng_key2, **kwargs):
        self.trans = trans
        self.inner_lr = kwargs.get("inner_lr", 0.01)
        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=kwargs.get("init_value", 0),
            peak_value=kwargs.get("peak_value", 1e-4),
            warmup_steps=kwargs.get("warmup_steps", 65),
            decay_steps=kwargs.get("decay_steps", 650),
            end_value=kwargs.get("end_value", 0),
        )
        tx = optimizer_class(scheduler_class)
        # Reconfigure for our data
        trans_params = self.trans.init(
            {"params": rng_key1, "dropout": rng_key2},
            jnp.zeros((10, 25, trans.state_dim)),
            jnp.zeros((10, 25, trans.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32),
        )
        self._train_state = TrainState.create(
            params=trans_params, tx=tx, apply_fn=self.trans.apply
        )

    def evaluation(self, batch, rng_key):
        return _eval_mamlp_step(self._train_state, self.inner_lr, batch, rng_key)

    def train(self, batch, rng_key):
        self._train_state, metrics = _train_mamlp_step(
            self._train_state, self.inner_lr, batch, rng_key
        )
        return metrics

    def get_params(self):
        return self._train_state.params


def maml_fit_task(state_fn, train_params, inner_lr, batch, rng_key):
    grad_fn = jax.grad(pref_loss_fn, argnums=1, has_aux=True)
    optx = optax.sgd(inner_lr)
    inner_state = TrainState.create(params=train_params, tx=optx, apply_fn=state_fn)
    grads, _ = grad_fn(inner_state.apply_fn, inner_state.params, batch, True, rng_key)
    inner_state = inner_state.apply_gradients(grads=grads)
    return inner_state


@jax.jit
def _eval_mamlp_step(state, inner_lr, batch, rng_key):
    def maml_loss(
        state_fn,
        train_params,
        inner_lr,
        t_sts,
        t_acts,
        t_ts,
        t_am,
        t_sts2,
        t_acts2,
        t_ts2,
        t_am2,
        t_l,
        v_sts,
        v_acts,
        v_ts,
        v_am,
        v_sts2,
        v_acts2,
        v_ts2,
        v_am2,
        v_l,
        rng_key1,
        rng_key2,
    ):
        train_batch = {
            "states": t_sts,
            "actions": t_acts,
            "timesteps": t_ts,
            "attn_mask": t_am,
            "states_2": t_sts2,
            "actions_2": t_acts2,
            "timesteps_2": t_ts2,
            "attn_mask_2": t_am2,
            "labels": t_l,
        }
        val_batch = {
            "states": v_sts,
            "actions": v_acts,
            "timesteps": v_ts,
            "attn_mask": v_am,
            "states_2": v_sts2,
            "actions_2": v_acts2,
            "timesteps_2": v_ts2,
            "attn_mask_2": v_am2,
            "labels": v_l,
        }
        updated_state = maml_fit_task(
            state_fn, train_params, inner_lr, train_batch, rng_key1
        )
        loss, acc = pref_loss_fn(
            updated_state.apply_fn, updated_state.params, val_batch, False, rng_key2
        )
        return loss, acc

    def task_loss(state_fn, train_params, inner_lr, rng_key):
        train_batch, val_batch = batch
        t_sts, t_acts, t_ts, t_am, t_sts2, t_acts2, t_ts2, t_am2, t_l = train_batch
        v_sts, v_acts, v_ts, v_am, v_sts2, v_acts2, v_ts2, v_am2, v_l = val_batch
        rng_keys1, rng_keys2 = jax.random.split(rng_key, (2, t_sts.shape[0]))
        p_losses, p_acc = jax.vmap(
            partial(maml_loss, state_fn, train_params, inner_lr)
        )(
            t_sts,
            t_acts,
            t_ts,
            t_am,
            t_sts2,
            t_acts2,
            t_ts2,
            t_am2,
            t_l,
            v_sts,
            v_acts,
            v_ts,
            v_am,
            v_sts2,
            v_acts2,
            v_ts2,
            v_am2,
            v_l,
            rng_keys1,
            rng_keys2,
        )
        return jnp.mean(p_losses), jnp.mean(p_acc)

    loss, acc = task_loss(state.apply_fn, state.params, inner_lr, rng_key)
    return dict(eval_loss=loss, eval_acc=acc)


@jax.jit
def _train_mamlp_step(state, inner_lr, batch, rng_key):
    def maml_loss(
        state_fn,
        train_params,
        inner_lr,
        t_sts,
        t_acts,
        t_ts,
        t_am,
        t_sts2,
        t_acts2,
        t_ts2,
        t_am2,
        t_l,
        v_sts,
        v_acts,
        v_ts,
        v_am,
        v_sts2,
        v_acts2,
        v_ts2,
        v_am2,
        v_l,
        rng_key1,
        rng_key2,
    ):
        train_batch = {
            "states": t_sts,
            "actions": t_acts,
            "timesteps": t_ts,
            "attn_mask": t_am,
            "states_2": t_sts2,
            "actions_2": t_acts2,
            "timesteps_2": t_ts2,
            "attn_mask_2": t_am2,
            "labels": t_l,
        }
        val_batch = {
            "states": v_sts,
            "actions": v_acts,
            "timesteps": v_ts,
            "attn_mask": v_am,
            "states_2": v_sts2,
            "actions_2": v_acts2,
            "timesteps_2": v_ts2,
            "attn_mask_2": v_am2,
            "labels": v_l,
        }
        updated_state = maml_fit_task(
            state_fn, train_params, inner_lr, train_batch, rng_key1
        )
        loss, acc = pref_loss_fn(
            updated_state.apply_fn, updated_state.params, val_batch, False, rng_key2
        )
        return loss, acc

    def task_loss(state_fn, train_params, inner_lr, rng_key):
        train_batch, val_batch = batch
        t_sts, t_acts, t_ts, t_am, t_sts2, t_acts2, t_ts2, t_am2, t_l = train_batch
        v_sts, v_acts, v_ts, v_am, v_sts2, v_acts2, v_ts2, v_am2, v_l = val_batch
        rng_keys1, rng_keys2 = jax.random.split(rng_key, (2, t_sts.shape[0]))
        p_losses, p_acc = jax.vmap(
            partial(maml_loss, state_fn, train_params, inner_lr)
        )(
            t_sts,
            t_acts,
            t_ts,
            t_am,
            t_sts2,
            t_acts2,
            t_ts2,
            t_am2,
            t_l,
            v_sts,
            v_acts,
            v_ts,
            v_am,
            v_sts2,
            v_acts2,
            v_ts2,
            v_am2,
            v_l,
            rng_keys1,
            rng_keys2,
        )
        return jnp.mean(p_losses), jnp.mean(p_acc)

    grad_fn = jax.value_and_grad(task_loss, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state.apply_fn, state.params, inner_lr, rng_key)
    new_train_state = state.apply_gradients(grads=grads)
    metrics = dict(training_loss=loss, training_acc=acc)
    return new_train_state, metrics


class InterventionMLPTrainer(object):

    def __init__(self, imlp, rng_key1, rng_key2, **kwargs):
        self.imlp = imlp

        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=kwargs.get("init_value", 0),
            peak_value=kwargs.get("peak_value", 1e-4),
            warmup_steps=kwargs.get("warmup_steps", 65),
            decay_steps=kwargs.get("decay_steps", 650),
            end_value=kwargs.get("end_value", 0),
        )

        tx = optimizer_class(scheduler_class)

        # Reconfigure for our data
        imlp_params = self.imlp.init(
            {"params": rng_key1, "dropout": rng_key2},
            jnp.zeros((10, imlp.observation_dim)),
        )
        self._train_states = TrainState.create(
            params=imlp_params, tx=tx, apply_fn=self.imlp.apply
        )

    def evaluation(self, batch, rng_key):
        return _eval_imlp_step(self._train_state, batch, rng_key)

    def train(self, batch, rng_key):
        self._train_state, metrics = _train_imlp_step(self._train_state, batch, rng_key)
        return metrics


@jax.jit
def _eval_imlp_step(state, batch, rng_key):
    loss, acc = imlp_loss_fn(state.apply_fn, state.params, batch, False, rng_key)
    return dict(eval_loss=loss, eval_acc=acc)


@jax.jit
def _train_imlp_step(state, batch, rng_key):
    grad_fn = jax.value_and_grad(imlp_loss_fn, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state.apply_fn, state.params, batch, True, rng_key)

    new_train_state = state.apply_gradients(grads=grads)
    metrics = dict(training_loss=loss, training_acc=acc)
    return new_train_state, metrics
