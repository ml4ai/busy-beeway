from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from ml_collections import ConfigDict
from transformers.training.jax_utils import (
    ad_loss_fn,
    af_loss_fn,
    pref_loss_fn,
    q_loss_fn,
    sd_loss_fn,
    sf_loss_fn,
    val_loss,
    actor_loss,
    q_loss,
)


class PrefTransformerTrainer(object):

    def __init__(self, trans, **kwargs):
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

        self.evaluation = nnx.cached_partial(
            _eval_pref_step,
            nnx.Optimizer(trans, tx),
        )

        self.train = nnx.cached_partial(
            _train_pref_step,
            nnx.Optimizer(trans, tx),
        )


@nnx.jit
def _eval_pref_step(state, batch):
    loss, acc = pref_loss_fn(state.model, batch, False)
    return dict(eval_loss=loss, eval_acc=acc)


@nnx.jit
def _train_pref_step(state, batch):
    (loss, acc), grads = nnx.value_and_grad(pref_loss_fn, has_aux=True)(
        state.model, batch, True
    )
    state.update(grads)
    return dict(training_loss=loss, training_acc=acc)


class DecTransformerTrainer(object):
    # output_type (str) helps the trainer know which to loss function to use, == "Q" (state-action values),
    # == "S_D" (discrete states), == "S_F" (feature-based states),
    # == "A_D" (discete actions), == "A_F" (feature-based actions).
    # S_F/A_F uses l2 loss regardless of whether features are continuous or discrete
    def __init__(
        self,
        dec,
        output_type="A_F",
        **kwargs,
    ):
        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=kwargs.get("init_value", 0),
            peak_value=kwargs.get("peak_value", 1e-4),
            warmup_steps=kwargs.get("warmup_steps", 65),
            decay_steps=kwargs.get("decay_steps", 650),
            end_value=kwargs.get("end_value", 0),
        )
        tx = optax.chain(
            optax.clip_by_global_norm(0.25), optimizer_class(scheduler_class)
        )
        # Reconfigure for our data

        match output_type:
            case "Q":
                self.train = nnx.cached_partial(
                    _train_Qdec_step, nnx.Optimizer(dec, tx)
                )
            case "S_D":
                self.train = nnx.cached_partial(
                    _train_S_Ddec_step, nnx.Optimizer(dec, tx)
                )
            case "S_F":
                self.train = nnx.cached_partial(
                    _train_S_Fdec_step, nnx.Optimizer(dec, tx)
                )
            case "A_D":
                self.train = nnx.cached_partial(
                    _train_A_Ddec_step, nnx.Optimizer(dec, tx)
                )
            case _:
                self.train = nnx.cached_partial(
                    _train_A_Fdec_step, nnx.Optimizer(dec, tx)
                )


@nnx.jit
def _train_Qdec_step(state, batch):
    loss, grads = nnx.value_and_grad(q_loss_fn)(state.model, batch, True)
    state.update(grads)
    return dict(training_loss=loss)


@nnx.jit
def _train_S_Ddec_step(state, batch):
    (loss, acc), grads = nnx.value_and_grad(sd_loss_fn, has_aux=True)(
        state.model, batch, True
    )
    state.update(grads)
    return dict(training_loss=loss, training_acc=acc)


@nnx.jit
def _train_S_Fdec_step(state, batch):
    loss, grads = nnx.value_and_grad(sf_loss_fn)(state.model, batch, True)
    state.update(grads)
    return dict(training_loss=loss)


@nnx.jit
def _train_A_Ddec_step(state, batch):
    (loss, acc), grads = nnx.value_and_grad(ad_loss_fn, has_aux=True)(
        state.model, batch, True
    )
    state.update(grads)
    return dict(training_loss=loss, training_acc=acc)


@nnx.jit
def _train_A_Fdec_step(state, batch):
    loss, grads = nnx.value_and_grad(af_loss_fn)(state.model, batch, True)
    state.update(grads)
    return dict(training_loss=loss)


def breakpoint_if_nonfinite(x):
  is_finite = jnp.isfinite(x).all()
  def true_fn(x):
    pass
  def false_fn(x):
    jax.debug.breakpoint()
  lax.cond(is_finite, true_fn, false_fn, x)

class IQLTrainer(object):

    def __init__(self, actor, vCritic, qCritic, tCritic, **kwargs):
        opt_decay_schedule = kwargs.get("opt_decay_schedule", "cosine")

        actor_lr = kwargs.get("actor_lr", 3e-4)
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(
                actor_lr, kwargs.get("max_steps", int(1e6))
            )
            actor_optimizer = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            actor_optimizer = optax.adam(learning_rate=actor_lr)

        vCritic_optimizer = optax.adam(learning_rate=kwargs.get("value_lr", 3e-4))

        qCritic_optimizer = optax.adam(learning_rate=kwargs.get("critic_lr", 3e-4))

        self.train = nnx.cached_partial(
            _train_IQL_step,
            nnx.Optimizer(actor, actor_optimizer),
            nnx.Optimizer(vCritic, vCritic_optimizer),
            nnx.Optimizer(qCritic, qCritic_optimizer),
            tCritic,
            kwargs.get("expectile", 0.8),
            kwargs.get("temperature", 0.1),
            kwargs.get("discount", 0.99),
            kwargs.get("tau", 0.005),
        )


@nnx.jit
def _train_IQL_step(
    actor_state, v_state, q_state, tCritic, expectile, temperature, discount, tau, batch
):
    v_loss, v_grads = nnx.value_and_grad(val_loss)(
        v_state.model, tCritic, expectile, batch
    )
    breakpoint_if_nonfinite(v_loss)
    v_state.update(v_grads)

    act_loss, act_grads = nnx.value_and_grad(actor_loss)(
        actor_state.model, v_state.model, tCritic, temperature, batch
    )
    breakpoint_if_nonfinite(act_loss)
    actor_state.update(act_grads)

    qq_loss, qq_grads = nnx.value_and_grad(q_loss)(
        q_state.model, v_state.model, discount, batch
    )
    breakpoint_if_nonfinite(qq_loss)
    q_state.update(qq_grads)

    t_p_state = nnx.state(tCritic, nnx.Param)
    q_p_state = nnx.state(q_state.model, nnx.Param)
    new_t_p_state = jax.tree.map(
        lambda x, y: y * tau + x * (1 - tau), t_p_state, q_p_state
    )
    nnx.update(tCritic, new_t_p_state)
    jax.debug.print("{loss}", loss=v_loss + act_loss + qq_loss)
    return dict(training_loss=v_loss + act_loss + qq_loss)


# class MentorTransformerTrainer(object):

#     def __init__(self, trans, rng_key1, rng_key2, pretrained_params=None, **kwargs):
#         self.trans = trans

#         optimizer_class = optax.adamw
#         # May need to reconfigure for our data
#         scheduler_class = optax.warmup_cosine_decay_schedule(
#             init_value=kwargs.get("init_value", 0),
#             peak_value=kwargs.get("peak_value", 1e-4),
#             warmup_steps=kwargs.get("warmup_steps", 65),
#             decay_steps=kwargs.get("decay_steps", 650),
#             end_value=kwargs.get("end_value", 0),
#         )
#         tx = optimizer_class(scheduler_class)
#         # Reconfigure for our data
#         if pretrained_params is None:
#             trans_params = self.trans.init(
#                 {"params": rng_key1, "dropout": rng_key2},
#                 jnp.zeros((10, 25, trans.state_dim)),
#                 jnp.zeros((10, 25, trans.action_dim)),
#                 jnp.ones((10, 25), dtype=jnp.int32),
#             )
#             self._train_state = TrainState.create(
#                 params=trans_params, tx=tx, apply_fn=self.trans.apply
#             )
#         else:
#             self._train_state = TrainState.create(
#                 params=pretrained_params, tx=tx, apply_fn=self.trans.apply
#             )

#     def evaluation(self, batch, rng_key):
#         return _eval_mentor_step(self._train_state, batch, rng_key)

#     def train(self, batch, rng_key):
#         self._train_state, metrics = _train_mentor_step(
#             self._train_state, batch, rng_key
#         )
#         return metrics

#     def get_params(self):
#         return self._train_state.params


# @jax.jit
# def _eval_mentor_step(state, batch, rng_key):
#     loss, acc = mentor_loss_fn(state.apply_fn, state.params, batch, False, rng_key)
#     return dict(eval_loss=loss, eval_acc=acc)


# @jax.jit
# def _train_mentor_step(state, batch, rng_key):
#     grad_fn = jax.value_and_grad(mentor_loss_fn, argnums=1, has_aux=True)
#     (loss, acc), grads = grad_fn(state.apply_fn, state.params, batch, True, rng_key)
#     new_train_state = state.apply_gradients(grads=grads)
#     metrics = dict(training_loss=loss, training_acc=acc)
#     return new_train_state, metrics


# class MAMLPTTrainer(object):

#     def __init__(self, trans, rng_key1, rng_key2, **kwargs):
#         self.trans = trans
#         self.inner_lr = kwargs.get("inner_lr", 0.01)
#         optimizer_class = optax.adamw
#         # May need to reconfigure for our data
#         scheduler_class = optax.warmup_cosine_decay_schedule(
#             init_value=kwargs.get("init_value", 0),
#             peak_value=kwargs.get("peak_value", 1e-4),
#             warmup_steps=kwargs.get("warmup_steps", 65),
#             decay_steps=kwargs.get("decay_steps", 650),
#             end_value=kwargs.get("end_value", 0),
#         )
#         tx = optimizer_class(scheduler_class)
#         # Reconfigure for our data
#         trans_params = self.trans.init(
#             {"params": rng_key1, "dropout": rng_key2},
#             jnp.zeros((10, 25, trans.state_dim)),
#             jnp.zeros((10, 25, trans.action_dim)),
#             jnp.ones((10, 25), dtype=jnp.int32),
#         )
#         self._train_state = TrainState.create(
#             params=trans_params, tx=tx, apply_fn=self.trans.apply
#         )

#     def evaluation(self, batch, rng_key):
#         return _eval_mamlp_step(self._train_state, self.inner_lr, batch, rng_key)

#     def train(self, batch, rng_key):
#         self._train_state, metrics = _train_mamlp_step(
#             self._train_state, self.inner_lr, batch, rng_key
#         )
#         return metrics

#     def get_params(self):
#         return self._train_state.params


# def maml_fit_task(state_fn, train_params, inner_lr, batch, rng_key):
#     grad_fn = jax.grad(pref_loss_fn, argnums=1, has_aux=True)
#     optx = optax.sgd(inner_lr)
#     inner_state = TrainState.create(params=train_params, tx=optx, apply_fn=state_fn)
#     grads, _ = grad_fn(inner_state.apply_fn, inner_state.params, batch, True, rng_key)
#     inner_state = inner_state.apply_gradients(grads=grads)
#     return inner_state


# @jax.jit
# def _eval_mamlp_step(state, inner_lr, batch, rng_key):
#     def maml_loss(
#         state_fn,
#         train_params,
#         inner_lr,
#         t_sts,
#         t_acts,
#         t_ts,
#         t_am,
#         t_sts2,
#         t_acts2,
#         t_ts2,
#         t_am2,
#         t_l,
#         v_sts,
#         v_acts,
#         v_ts,
#         v_am,
#         v_sts2,
#         v_acts2,
#         v_ts2,
#         v_am2,
#         v_l,
#         rng_key1,
#         rng_key2,
#     ):
#         train_batch = {
#             "states": t_sts,
#             "actions": t_acts,
#             "timesteps": t_ts,
#             "attn_mask": t_am,
#             "states_2": t_sts2,
#             "actions_2": t_acts2,
#             "timesteps_2": t_ts2,
#             "attn_mask_2": t_am2,
#             "labels": t_l,
#         }
#         val_batch = {
#             "states": v_sts,
#             "actions": v_acts,
#             "timesteps": v_ts,
#             "attn_mask": v_am,
#             "states_2": v_sts2,
#             "actions_2": v_acts2,
#             "timesteps_2": v_ts2,
#             "attn_mask_2": v_am2,
#             "labels": v_l,
#         }
#         updated_state = maml_fit_task(
#             state_fn, train_params, inner_lr, train_batch, rng_key1
#         )
#         loss, acc = pref_loss_fn(
#             updated_state.apply_fn, updated_state.params, val_batch, False, rng_key2
#         )
#         return loss, acc

#     def task_loss(state_fn, train_params, inner_lr, rng_key):
#         train_batch, val_batch = batch
#         t_sts, t_acts, t_ts, t_am, t_sts2, t_acts2, t_ts2, t_am2, t_l = train_batch
#         v_sts, v_acts, v_ts, v_am, v_sts2, v_acts2, v_ts2, v_am2, v_l = val_batch
#         rng_keys1, rng_keys2 = jax.random.split(rng_key, (2, t_sts.shape[0]))
#         p_losses, p_acc = jax.vmap(
#             partial(maml_loss, state_fn, train_params, inner_lr)
#         )(
#             t_sts,
#             t_acts,
#             t_ts,
#             t_am,
#             t_sts2,
#             t_acts2,
#             t_ts2,
#             t_am2,
#             t_l,
#             v_sts,
#             v_acts,
#             v_ts,
#             v_am,
#             v_sts2,
#             v_acts2,
#             v_ts2,
#             v_am2,
#             v_l,
#             rng_keys1,
#             rng_keys2,
#         )
#         return jnp.mean(p_losses), jnp.mean(p_acc)

#     loss, acc = task_loss(state.apply_fn, state.params, inner_lr, rng_key)
#     return dict(eval_loss=loss, eval_acc=acc)


# @jax.jit
# def _train_mamlp_step(state, inner_lr, batch, rng_key):
#     def maml_loss(
#         state_fn,
#         train_params,
#         inner_lr,
#         t_sts,
#         t_acts,
#         t_ts,
#         t_am,
#         t_sts2,
#         t_acts2,
#         t_ts2,
#         t_am2,
#         t_l,
#         v_sts,
#         v_acts,
#         v_ts,
#         v_am,
#         v_sts2,
#         v_acts2,
#         v_ts2,
#         v_am2,
#         v_l,
#         rng_key1,
#         rng_key2,
#     ):
#         train_batch = {
#             "states": t_sts,
#             "actions": t_acts,
#             "timesteps": t_ts,
#             "attn_mask": t_am,
#             "states_2": t_sts2,
#             "actions_2": t_acts2,
#             "timesteps_2": t_ts2,
#             "attn_mask_2": t_am2,
#             "labels": t_l,
#         }
#         val_batch = {
#             "states": v_sts,
#             "actions": v_acts,
#             "timesteps": v_ts,
#             "attn_mask": v_am,
#             "states_2": v_sts2,
#             "actions_2": v_acts2,
#             "timesteps_2": v_ts2,
#             "attn_mask_2": v_am2,
#             "labels": v_l,
#         }
#         updated_state = maml_fit_task(
#             state_fn, train_params, inner_lr, train_batch, rng_key1
#         )
#         loss, acc = pref_loss_fn(
#             updated_state.apply_fn, updated_state.params, val_batch, False, rng_key2
#         )
#         return loss, acc

#     def task_loss(state_fn, train_params, inner_lr, rng_key):
#         train_batch, val_batch = batch
#         t_sts, t_acts, t_ts, t_am, t_sts2, t_acts2, t_ts2, t_am2, t_l = train_batch
#         v_sts, v_acts, v_ts, v_am, v_sts2, v_acts2, v_ts2, v_am2, v_l = val_batch
#         rng_keys1, rng_keys2 = jax.random.split(rng_key, (2, t_sts.shape[0]))
#         p_losses, p_acc = jax.vmap(
#             partial(maml_loss, state_fn, train_params, inner_lr)
#         )(
#             t_sts,
#             t_acts,
#             t_ts,
#             t_am,
#             t_sts2,
#             t_acts2,
#             t_ts2,
#             t_am2,
#             t_l,
#             v_sts,
#             v_acts,
#             v_ts,
#             v_am,
#             v_sts2,
#             v_acts2,
#             v_ts2,
#             v_am2,
#             v_l,
#             rng_keys1,
#             rng_keys2,
#         )
#         return jnp.mean(p_losses), jnp.mean(p_acc)

#     grad_fn = jax.value_and_grad(task_loss, argnums=1, has_aux=True)
#     (loss, acc), grads = grad_fn(state.apply_fn, state.params, inner_lr, rng_key)
#     new_train_state = state.apply_gradients(grads=grads)
#     metrics = dict(training_loss=loss, training_acc=acc)
#     return new_train_state, metrics

# class ValTransformerTrainer(object):
#     def __init__(
#         self,
#         val,
#         rng_key1,
#         rng_key2,
#         pretrained_params=None,
#         **kwargs,
#     ):
#         self.val = val
#         optimizer_class = optax.adamw
#         # May need to reconfigure for our data
#         scheduler_class = optax.warmup_cosine_decay_schedule(
#             init_value=kwargs.get("init_value", 0),
#             peak_value=kwargs.get("peak_value", 1e-4),
#             warmup_steps=kwargs.get("warmup_steps", 65),
#             decay_steps=kwargs.get("decay_steps", 650),
#             end_value=kwargs.get("end_value", 0),
#         )
#         tx = optimizer_class(scheduler_class)
#         # Reconfigure for our data
#         if pretrained_params is None:
#             val_params = self.val.init(
#                 {"params": rng_key1, "dropout": rng_key2},
#                 jnp.zeros((10, 25, val.state_dim)),
#                 jnp.ones((10, 25), dtype=jnp.int32),
#             )
#             self._train_state = TrainState.create(
#                 params=val_params, tx=tx, apply_fn=self.val.apply
#             )
#         else:
#             self._train_state = TrainState.create(
#                 params=pretrained_params, tx=tx, apply_fn=self.val.apply
#             )

#     def evaluation(self, batch, rng_key):
#         return _eval_val_step(self._train_state, batch, rng_key)

#     def train(self, batch, rng_key):
#         self._train_state, metrics = _train_val_step(self._train_state, batch, rng_key)
#         return metrics

#     def get_params(self):
#         return self._train_state.params
