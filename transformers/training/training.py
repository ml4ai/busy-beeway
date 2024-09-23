import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from transformers.training.jax_utils import pref_loss_fn


class PrefTransformerTrainer(object):

    def __init__(self, trans, rng_key1, rng_key2, **kwargs):
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
        trans_params = self.trans.init(
            {"params": rng_key1, "dropout": rng_key2},
            jnp.zeros((10, 25, trans.observation_dim)),
            jnp.ones((10, 25)),
        )
        self._train_state = TrainState.create(
            params=trans_params, tx=tx, apply_fn=self.trans.apply
        )

    def evaluation(self, batch, rng_key):
        return _eval_pref_step(self._train_state, batch, rng_key)

    def train(self, batch, rng_key):
        self._train_state, metrics = _train_pref_step(self._train_state, batch, rng_key)
        return metrics


@jax.jit
def _eval_pref_step(state, batch, rng_key):
    loss, acc = pref_loss_fn(state.apply_fn, state.params, batch, False, rng_key)
    return dict(eval_loss=loss, eval_acc=acc)


@jax.jit
def _train_pref_step(state, batch, rng_key):
    grad_fn = jax.value_and_grad(pref_loss_fn, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state.apply_fn, state.params, batch, True, rng_key)
    jax.debug.print("gradients: {x}",x=grads)
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
