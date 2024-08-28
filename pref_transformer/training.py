from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from jax_utils import next_rng, pref_loss_fn


class PrefTransformer(object):

    def __init__(self, trans, observation_dim):
        self.trans = trans
        self.observation_dim = observation_dim

        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-4,
            warmup_steps=65,
            decay_steps=650,
            end_value=0,
        )

        tx = optimizer_class(scheduler_class)

        # Reconfigure for our data
        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.ones((10, 25), dtype=jnp.int32),
        )
        self._train_state = TrainState.create(
            params=trans_params, tx=tx, apply_fn=self.trans.apply
        )

    def evaluation(self, batch):
        return _eval_pref_step(self._train_state, batch, next_rng())

    def train(self, batch):
        self._train_state, metrics = _train_pref_step(
            self._train_state, batch, next_rng()
        )
        return metrics


@jax.jit
def _eval_pref_step(state, batch, rng):
    loss = pref_loss_fn(state, state.params, batch, rng)
    return dict(eval_loss=loss)


@jax.jit
def _train_pref_step(state, batch, rng):
    grad_fn = jax.value_and_grad(pref_loss_fn, argnums=1)
    loss, grads = grad_fn(state, state.params, batch, rng)

    new_train_state = state.apply_gradients(grads=grads)
    metrics = dict(training_loss=loss)
    return new_train_state, metrics


class intervention_MLP(object):

    def __init__(self, imlp, observation_dim):
        self.imlp = imlp
        self.observation_dim = observation_dim

        optimizer_class = optax.adamw
        # May need to reconfigure for our data
        scheduler_class = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-4,
            warmup_steps=650,
            decay_steps=6500,
            end_value=0,
        )

        tx = optimizer_class(scheduler_class)

        # Reconfigure for our data
        imlp_params = self.imlp.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, self.observation_dim)),
        )
        self._train_states = TrainState.create(params=imlp_params, tx=tx, apply_fn=None)

    def evaluation(self, batch):
        return _eval_imlp_step(self._train_state, batch, next_rng())

    def train(self, batch):
        self._train_state, metrics = _train_imlp_step(
            self._train_state, batch, next_rng()
        )
        return metrics


@jax.jit
def _eval_imlp_step(state, batch, rng):
    loss, acc = imlp_loss_fn(state, state.params, batch, rng)
    return dict(eval_loss=loss, eval_acc=acc)


@jax.jit
def _train_imlp_step(state, batch, rng):
    grad_fn = jax.value_and_grad(imlp_loss_fn, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch, rng)

    new_train_state = state.apply_gradients(grads=grads)
    metrics = dict(training_loss=loss, training_acc=acc)
    return new_train_state, metrics
