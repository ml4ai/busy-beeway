import numpy as np
import jax
import jax.numpy as jnp
import optax


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


def cross_ent_loss(logits, target):

    if len(target.shape) == 1:
        label = jax.nn.one_hot(target, num_classes=2)
    else:
        label = target

    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=label))
    return loss


def kld_loss(p, q):
    return jnp.mean(
        jnp.sum(jnp.where(p != 0, p * (jnp.log(p) - jnp.log(q)), 0), axis=-1)
    )


def custom_softmax(array, axis=-1, temperature=1.0):
    array = array / temperature
    return jax.nn.softmax(array, axis=axis)


def pref_accuracy(logits, target):
    predicted_class = jnp.argmax(logits, axis=1)
    target_class = jnp.argmax(target, axis=1)
    return jnp.mean(predicted_class == target_class)


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


def pref_loss_fn(state, train_params, batch, rng):
    obs_1 = batch["observations"]
    obs_2 = batch["observations_2"]
    timestep_1 = batch["timesteps"]
    timestep_2 = batch["timesteps_2"]
    am_1 = batch["attn_mask"]
    am_2 = batch["attn_mask_2"]
    labels = batch["labels"]

    B, T, _ = batch["observations"].shape

    rng, _ = jax.random.split(rng)

    trans_pred_1, _ = state.apply_fn(
        train_params,
        obs_1,
        timestep_1,
        training=False,
        attn_mask=am_1,
        rngs={"dropout": rng},
    )
    trans_pred_2, _ = state.apply_fn(
        train_params,
        obs_2,
        timestep_2,
        training=False,
        attn_mask=am_2,
        rngs={"dropout": rng},
    )

    trans_pred_1 = trans_pred_1["weighted_sum"]
    trans_pred_2 = trans_pred_2["weighted_sum"]

    sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
    sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

    logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)

    return cross_ent_loss(logits, labels)


def imlp_loss_fn(state, train_params, batch, rng):
    obs = batch["observations"]
    labels = batch["labels"]

    rng, _ = jax.random.split(rng)

    imlp_pred = state.apply_fn(
        train_params,
        obs,
        training=False,
        rngs={"dropout": rng},
    )

    rng, split_rng = jax.random.split(rng)

    """ reward function loss """
    pred_labels = (imlp_pred > 0).astype(jnp.float32)
    label_target = jax.lax.stop_gradient(labels)
    imlp_loss = optax.sigmoid_binary_cross_entropy(imlp_pred, label_target).mean()
    imlp_acc = (pred_labels == labels).mean()
    return imlp_loss, imlp_acc


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)