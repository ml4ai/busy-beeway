import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial


class JaxRNG(object):
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        return next_rng


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG(seed)


def next_rng():
    global jax_utils_rng
    return jax_utils_rng()


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


# classes only needs to be set if target is not a one hot vector, otherwise |logits| == |label| should already be true
def cross_ent_loss(logits, target, classes=2):

    if len(target.shape) == 1:
        label = jax.nn.one_hot(target, num_classes=classes)
    else:
        label = target
    loss = jnp.nanmean(optax.softmax_cross_entropy(logits=logits, labels=label))
    return loss


def kld_loss(p, q):
    return jnp.mean(
        jnp.sum(jnp.where(p != 0, p * (jnp.log(p) - jnp.log(q)), 0), axis=-1)
    )


def custom_softmax(array, axis=-1, temperature=1.0):
    array = array / temperature
    return jax.nn.softmax(array, axis=axis)


def pref_accuracy(logits, target):
    predicted_class = jnp.argmax(logits, axis=1) * 1.0
    predicted_class = jnp.where(
        jnp.isclose(logits[:, 0], logits[:, 1]), 0.5, predicted_class
    )
    return jnp.nanmean(predicted_class == target)


def dt_accuracy(logits, target):
    predicted_class = jnp.argmax(logits, axis=2) * 1.0
    true_target = jnp.argmax(target, axis=2) * 1.0
    return jnp.nanmean(predicted_class == true_target)


def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]

        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
        for i in range(n_outputs)
    )

    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)

    return multi_grad_fn


def pref_loss_fn(state_fn, train_params, batch, training, rng):
    sts_1 = batch["states"]
    sts_2 = batch["states_2"]
    acts_1 = batch["actions"]
    acts_2 = batch["actions_2"]
    timestep_1 = batch["timesteps"]
    timestep_2 = batch["timesteps_2"]
    am_1 = batch["attn_mask"]
    am_2 = batch["attn_mask_2"]
    labels = batch["labels"]

    B, T, _ = batch["states"].shape

    trans_pred_1, _ = state_fn(
        train_params,
        sts_1,
        acts_1,
        timestep_1,
        training=training,
        attn_mask=am_1,
        rngs={"dropout": rng},
    )
    trans_pred_2, _ = state_fn(
        train_params,
        sts_2,
        acts_2,
        timestep_2,
        training=training,
        attn_mask=am_2,
        rngs={"dropout": rng},
    )

    trans_pred_1 = trans_pred_1["weighted_sum"]
    trans_pred_2 = trans_pred_2["weighted_sum"]

    sum_pred_1 = jnp.nanmean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
    sum_pred_2 = jnp.nanmean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

    logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

    return cross_ent_loss(logits, labels), pref_accuracy(logits, labels)


def q_loss_fn(state_fn, train_params, batch, training, rng):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    Q_preds, _, _, _ = state_fn(
        train_params,
        rtns,
        sts,
        acts,
        timestep,
        training=training,
        attn_mask=am,
        rngs={"dropout": rng},
    )
    return jnp.nanmean(optax.l2_loss(predictions=Q_preds, targets=rtns))


def v_loss_fn(state_fn, train_params, batch, training, rng):
    sts = batch["states"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    V_preds = state_fn(
        train_params,
        sts,
        timestep,
        training=training,
        attn_mask=am,
        rngs={"dropout": rng},
    )
    return jnp.nanmean(optax.l2_loss(predictions=V_preds, targets=rtns))


def sd_loss_fn(state_fn, train_params, batch, training, rng):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, s_preds, _ = state_fn(
        train_params,
        rtns,
        sts,
        acts,
        timestep,
        training=training,
        attn_mask=am,
        rngs={"dropout": rng},
    )
    return cross_ent_loss(s_preds, sts), dt_accuracy(s_preds, sts)


def sf_loss_fn(state_fn, train_params, batch, training, rng):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, s_preds, _ = state_fn(
        train_params,
        rtns,
        sts,
        acts,
        timestep,
        training=training,
        attn_mask=am,
        rngs={"dropout": rng},
    )
    return jnp.nanmean(optax.l2_loss(predictions=s_preds, targets=sts))


def ad_loss_fn(state_fn, train_params, batch, training, rng):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, _, a_preds = state_fn(
        train_params,
        rtns,
        sts,
        acts,
        timestep,
        training=training,
        attn_mask=am,
        rngs={"dropout": rng},
    )
    return cross_ent_loss(a_preds, acts), dt_accuracy(a_preds, acts)


def af_loss_fn(state_fn, train_params, batch, training, rng):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, _, a_preds = state_fn(
        train_params,
        rtns,
        sts,
        acts,
        timestep,
        training=training,
        attn_mask=am,
        rngs={"dropout": rng},
    )
    return jnp.nanmean(optax.l2_loss(predictions=a_preds, targets=acts))


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)
