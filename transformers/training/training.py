from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from ml_collections import ConfigDict
from transformers.training.jax_utils import pt_loss_fn, mr_loss_fn


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
            _eval_pt_step,
            nnx.Optimizer(trans, tx),
        )

        self.train = nnx.cached_partial(
            _train_pt_step,
            nnx.Optimizer(trans, tx),
        )


@nnx.jit
def _eval_pt_step(state, batch):
    loss, acc = pt_loss_fn(state.model, batch, False)
    return dict(eval_loss=loss, eval_acc=acc)


@nnx.jit
def _train_pt_step(state, batch):
    (loss, acc), grads = nnx.value_and_grad(pt_loss_fn, has_aux=True)(
        state.model, batch, True
    )
    state.update(grads)
    return dict(training_loss=loss, training_acc=acc)


class MRTrainer(object):

    def __init__(self, qmlp, **kwargs):
        optimizer_class = optax.adam
        tx = optimizer_class(kwargs.get("lr", 3e-4))

        self.evaluation = nnx.cached_partial(
            _eval_mr_step,
            nnx.Optimizer(qmlp, tx),
        )

        self.train = nnx.cached_partial(
            _train_mr_step,
            nnx.Optimizer(qmlp, tx),
        )


@nnx.jit
def _eval_mr_step(state, batch):
    loss, acc = mr_loss_fn(state.model, batch, False)
    return dict(eval_loss=loss, eval_acc=acc)


@nnx.jit
def _train_mr_step(state, batch):
    (loss, acc), grads = nnx.value_and_grad(mr_loss_fn, has_aux=True)(
        state.model, batch, True
    )
    state.update(grads)
    return dict(training_loss=loss, training_acc=acc)
