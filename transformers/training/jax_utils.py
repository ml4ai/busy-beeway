from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


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
        label = jnp.where(jnp.stack([jnp.all(label == 0,axis=1), jnp.all(label == 0,axis=1)]).T,0.5,label)
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


def mt_accuracy(logits, target):
    predicted_class = (logits > 0).astype(int)
    return jnp.nanmean(predicted_class == target)


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


def pref_loss_fn(model, batch, training):
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

    trans_pred_1, _ = model(
        sts_1,
        acts_1,
        timestep_1,
        am_1,
        training=training,
    )
    trans_pred_2, _ = model(
        sts_2,
        acts_2,
        timestep_2,
        am_2,
        training=training,
    )

    trans_pred_1 = trans_pred_1["weighted_sum"]
    trans_pred_2 = trans_pred_2["weighted_sum"]

    sum_pred_1 = jnp.nanmean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
    sum_pred_2 = jnp.nanmean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

    logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

    return cross_ent_loss(logits, labels), pref_accuracy(logits, labels)


# def mentor_loss_fn(model, batch, training):
#     sts_1 = batch["states"]
#     acts_1 = batch["actions"]
#     timestep_1 = batch["timesteps"]
#     am_1 = batch["attn_mask"]
#     labels = batch["labels"]

#     B, T, _ = batch["states"].shape

#     trans_pred_1, _ = model(
#         train_params,
#         sts_1,
#         acts_1,
#         timestep_1,
#         training=training,
#         attn_mask=am_1,
#         rngs={"dropout": rng},
#     )

#     trans_pred_1 = trans_pred_1["weighted_sum"]

#     sum_pred_1 = jnp.nanmean(trans_pred_1.reshape(B, T), axis=1)

#     return jnp.nanmean(
#         optax.sigmoid_binary_cross_entropy(logits=sum_pred_1, labels=labels)
#     ), mt_accuracy(sum_pred_1, labels)


def q_loss_fn(model, batch, training):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    Q_preds, _, _ = model(
        rtns,
        sts,
        acts,
        timestep,
        am,
        training=training,
    )
    return jnp.nanmean(optax.l2_loss(predictions=Q_preds, targets=rtns))


# def v_loss_fn(state_fn, train_params, batch, training, rng):
#     sts = batch["states"]
#     timestep = batch["timesteps"]
#     am = batch["attn_mask"]
#     B, T, _ = sts.shape
#     rtns = batch["returns"].reshape(B, T, 1)

#     V_preds = state_fn(
#         train_params,
#         sts,
#         timestep,
#         training=training,
#         attn_mask=am,
#         rngs={"dropout": rng},
#     )
#     return jnp.nanmean(optax.l2_loss(predictions=V_preds, targets=rtns))


def sd_loss_fn(model, batch, training):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, s_preds, _ = model(
        rtns,
        sts,
        acts,
        timestep,
        am,
        training=training,
    )
    return cross_ent_loss(s_preds, sts), dt_accuracy(s_preds, sts)


def sf_loss_fn(model, batch, training):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, s_preds, _ = model(
        rtns,
        sts,
        acts,
        timestep,
        am,
        training=training,
    )
    return jnp.nanmean(optax.l2_loss(predictions=s_preds, targets=sts))


def ad_loss_fn(model, batch, training):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, _, a_preds = model(
        rtns,
        sts,
        acts,
        timestep,
        am,
        training=training,
    )
    return cross_ent_loss(a_preds, acts), dt_accuracy(a_preds, acts)


def af_loss_fn(model, batch, training):
    sts = batch["states"]
    acts = batch["actions"]
    timestep = batch["timesteps"]
    am = batch["attn_mask"]
    B, T, _ = sts.shape
    rtns = batch["returns"].reshape(B, T, 1)

    _, _, a_preds = model(
        rtns,
        sts,
        acts,
        timestep,
        am,
        training=training,
    )
    return jnp.nanmean(optax.l2_loss(predictions=a_preds, targets=acts))


def vgrad(f, x):
    y, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


@nnx.vmap(in_axes=(None, 2, 2), out_axes=0)
def calculate_w_loss(model, x_1, x_2):
    x_1_s = model(x_1.T)
    x_2_s = model(x_2.T)
    return jnp.mean(jnp.mean(x_1_s, axis=0) - jnp.mean(x_2_s, axis=0))


def wasserstein_inner_loss_fn_gp(
    model,
    batch,
    output_dim,
    penalty_coeff,
    rngs: nnx.Rngs,
):
    @nnx.split_rngs(splits=output_dim)
    @nnx.vmap(in_axes=(None, 2, 2, 0), out_axes=0)
    def compute_gradient_penalty(model, samples_p, samples_q, rngs: nnx.Rngs):
        rng = rngs()
        eps = jax.random.uniform(rng, (samples_p.shape[1], 1))
        X = eps * samples_p.T + (1 - eps) * samples_q.T

        return ((jnp.linalg.norm(vgrad(model, X), 2, 1) - 1) ** 2).mean()

    objective = -(
        calculate_w_loss(model, batch["nnet_samples"], batch["gp_samples"]).sum()
    ) + (
        penalty_coeff
        * compute_gradient_penalty(
            model, batch["nnet_samples"], batch["gp_samples"], rngs
        ).sum()
    )
    return objective


def wasserstein_inner_loss_fn_lp(
    model,
    batch,
    output_dim,
    penalty_coeff,
    rngs: nnx.Rngs,
):
    @nnx.split_rngs(splits=output_dim)
    @nnx.vmap(in_axes=(None, 2, 2, 0), out_axes=0)
    def compute_gradient_penalty(model, samples_p, samples_q, rngs: nnx.Rngs):
        rng = rngs()
        eps = jax.random.uniform(rng, (samples_p.shape[1], 1))
        X = eps * samples_p.T + (1 - eps) * samples_q.T

        return (
            (
                jnp.clip(
                    jnp.linalg.norm(vgrad(model, X), 2, 1) - 1,
                    0.0,
                    jnp.inf,
                )
            )
            ** 2
        ).mean()

    objective = -(
        calculate_w_loss(model, batch["nnet_samples"], batch["gp_samples"]).sum()
    ) + (
        penalty_coeff
        * compute_gradient_penalty(
            model, batch["nnet_samples"], batch["gp_samples"], rngs
        ).sum()
    )
    return objective


def wasserstein_inner_loss_fn_nc(
    model,
    batch,
):
    objective = -(
        calculate_w_loss(model, batch["nnet_samples"], batch["gp_samples"]).sum()
    )
    return objective


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)
