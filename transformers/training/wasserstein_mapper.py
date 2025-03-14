import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from transformers.training.jax_utils import (
    calculate_w_loss,
    wasserstein_inner_loss_fn_gp,
    wasserstein_inner_loss_fn_lp,
    wasserstein_inner_loss_fn_nc,
)
from transformers.training.utils import ensure_dir, prng_to_raw, raw_to_prng


class LipschitzFunction(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(
            dim,
            200,
            kernel_init=nnx.initializers.xavier_normal(),
            bias_init=nnx.initializers.normal(),
            rngs=rngs,
        )
        self.lin2 = nnx.Linear(
            200,
            200,
            kernel_init=nnx.initializers.xavier_normal(),
            bias_init=nnx.initializers.normal(),
            rngs=rngs,
        )
        self.lin3 = nnx.Linear(
            200,
            1,
            kernel_init=nnx.initializers.xavier_normal(),
            bias_init=nnx.initializers.normal(),
            rngs=rngs,
        )

    def __call__(self, x):
        x = jnp.float32(x)
        x = self.lin1(x)
        x = nnx.softplus(x)
        x = self.lin2(x)
        x = nnx.softplus(x)
        x = self.lin3(x)
        return x


class WassersteinDistance:
    def __init__(
        self,
        bnn,
        gp,
        lipschitz_f_dim,
        output_dim,
        lipschitz_constraint_type="gp",
        wasserstein_lr=0.01,
        rngs=nnx.Rngs(0, params=1, dropout=2),
    ):
        self.bnn = bnn
        self.gp = gp
        self.output_dim = output_dim
        self.lipschitz_constraint_type = lipschitz_constraint_type
        assert self.lipschitz_constraint_type in ["gp", "lp", None]

        self.lipschitz_f = LipschitzFunction(dim=lipschitz_f_dim, rngs=rngs)

        if lipschitz_constraint_type == "gp":
            self.train_step = nnx.cached_partial(
                _train_w_step_gp,
                nnx.Optimizer(self.lipschitz_f, optax.adagrad(wasserstein_lr)),
                output_dim,
                10,
                rngs,
            )

        elif lipschitz_constraint_type == "lp":
            self.train_step = nnx.cached_partial(
                _train_w_step_lp,
                nnx.Optimizer(self.lipschitz_f, optax.adagrad(wasserstein_lr)),
                output_dim,
                10,
                rngs,
            )
        else:
            self.train_step = nnx.cached_partial(
                _train_w_step_nc,
                nnx.Optimizer(self.lipschitz_f, optax.adagrad(wasserstein_lr)),
            )

    def wasserstein_optimisation(
        self,
        X,
        n_samples,
        n_steps=10,
        threshold=None,
    ):
        n_samples_bag = n_samples * 1

        # Draw functions from GP
        gp_samples_bag = jnp.float32(
            self.gp.sample_functions(jnp.float64(X), n_samples_bag)
        )
        if self.output_dim > 1:
            gp_samples_bag = gp_samples_bag.squeeze()

        # Draw functions from Bayesian Neural network
        nnet_samples_bag = jnp.float32(self.bnn.sample_functions(X, n_samples_bag))
        if self.output_dim > 1:
            nnet_samples_bag = nnet_samples_bag.squeeze()

        batch = {"nnet_samples": nnet_samples_bag, "gp_samples": gp_samples_bag}

        for i in range(n_steps):
            loss, grads = self.train_step(batch)
            if threshold is not None:
                # Gradient Norm
                grad_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0], 2)

            if self.lipschitz_constraint_type is None:
                p_state = nnx.state(self.lipschitz_f, nnx.Param)
                new_p_state = jax.tree.map(lambda x: jnp.clip(x, -0.1, 0.1), p_state)
                nnx.update(self.lipschitz_f, new_p_state)
            if threshold is not None and grad_norm < threshold:
                print("WARNING: Grad norm (%.3f) lower than threshold (%.3f). ", end="")
                print("Stopping optimization at step %d" % (i))
                break


@partial(nnx.jit, static_argnums=1)
def _train_w_step_gp(state, output_dim, penalty_coeff, rngs, batch):
    loss, grads = nnx.value_and_grad(wasserstein_inner_loss_fn_gp)(
        state.model, batch, output_dim, penalty_coeff, rngs
    )
    state.update(grads)
    return loss, grads


@partial(nnx.jit, static_argnums=1)
def _train_w_step_lp(state, output_dim, penalty_coeff, rngs, batch):
    loss, grads = nnx.value_and_grad(wasserstein_inner_loss_fn_lp)(
        state.model, batch, output_dim, penalty_coeff, rngs
    )
    state.update(grads)
    return loss, grads


@nnx.jit
def _train_w_step_nc(state, batch):
    loss, grads = nnx.value_and_grad(wasserstein_inner_loss_fn_nc)(state.model, batch)
    state.update(grads)
    return loss, grads


class MapperWasserstein(object):
    def __init__(
        self,
        gp,
        bnn,
        data_generator,
        out_dir,
        input_dim=1,
        output_dim=1,
        n_data=256,
        wasserstein_steps=(200, 200),
        wasserstein_lr=0.01,
        wasserstein_thres=0.01,
        logger=None,
        lipschitz_constraint_type="gp",
        rngs=nnx.Rngs(0, params=1, dropout=2),
    ):
        self.gp = gp
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_dir = out_dir
        assert lipschitz_constraint_type in ["gp", "lp", None]
        self.lipschitz_constraint_type = lipschitz_constraint_type

        if type(wasserstein_steps) != list and type(wasserstein_steps) != tuple:
            wasserstein_steps = (wasserstein_steps, wasserstein_steps)
        self.wasserstein_steps = wasserstein_steps
        self.wasserstein_threshold = wasserstein_thres

        # Initialize the module of wasserstance distance
        self.wasserstein = WassersteinDistance(
            self.bnn,
            self.gp,
            self.n_data,
            output_dim=self.output_dim,
            wasserstein_lr=wasserstein_lr,
            lipschitz_constraint_type=self.lipschitz_constraint_type,
            rngs=rngs,
        )

        # Setup logger
        self.print_info = print if logger is None else logger.info

        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    def optimize(
        self,
        num_iters,
        n_samples=128,
        lr=1e-2,
        save_ckpt_every=50,
        print_every=10,
    ):
        wdist_hist = []

        wasserstein_steps = self.wasserstein_steps
        checkpointer = ocp.StandardCheckpointer()
        if self.output_dim > 1:
            train_step = nnx.cached_partial(
                _train_p_step_many,
                self.wasserstein.lipschitz_f,
                nnx.Optimizer(self.bnn, optax.rmsprop(lr)),
                n_samples,
            )
        else:
            train_step = nnx.cached_partial(
                _train_p_step_one,
                self.wasserstein.lipschitz_f,
                nnx.Optimizer(self.bnn, optax.rmsprop(lr)),
                n_samples,
            )
        # Prior loop
        for it in range(1, num_iters + 1):
            # Draw X
            X = self.data_generator.get(self.n_data)

            gp_samples = jnp.float32(
                self.gp.sample_functions(jnp.float64(X), n_samples)
            )
            if self.output_dim > 1:
                gp_samples = gp_samples.squeeze()
            # Optimisation of lipschitz_f
            self.wasserstein.wasserstein_optimisation(
                X,
                n_samples,
                n_steps=wasserstein_steps[1],
                threshold=self.wasserstein_threshold,
            )
            batch = {"gp_samples": gp_samples, "X": X}
            wdist, grads = train_step(batch)
            wdist_hist.append(float(wdist))
            if (it % print_every == 0) or it == 1:
                self.print_info(
                    ">>> Iteration # {:3d}: "
                    "Wasserstein Dist {:.4f}".format(it, float(wdist))
                )

            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                prng_to_raw(self.bnn)
                _, state = nnx.split(self.bnn)
                checkpointer.save(path, state)
                raw_to_prng(self.bnn)
                
        checkpointer.wait_until_finished()
        checkpointer.close()
        return wdist_hist


def calculate_w_loss_for_grad_one(bnn, lipschitz_f, n_samples, batch):
    nnet_samples = jnp.float32(bnn.sample_functions(batch["X"], n_samples))

    return calculate_w_loss(lipschitz_f, nnet_samples, batch["gp_samples"]).sum()


def calculate_w_loss_for_grad_many(bnn, lipschitz_f, n_samples, batch):
    nnet_samples = jnp.float32(bnn.sample_functions(batch["X"], n_samples))

    return calculate_w_loss(
        lipschitz_f, nnet_samples.squeeze(), batch["gp_samples"].squeeze()
    ).sum()


@partial(nnx.jit, static_argnums=2)
def _train_p_step_one(lipschitz_f, state, n_samples, batch):
    loss, grads = nnx.value_and_grad(calculate_w_loss_for_grad_one)(
        state.model, lipschitz_f, n_samples, batch
    )
    state.update(grads)
    return loss, grads


@partial(nnx.jit, static_argnums=2)
def _train_p_step_many(lipschitz_f, state, n_samples, batch):
    loss, grads = nnx.value_and_grad(calculate_w_loss_for_grad_many)(
        state.model, lipschitz_f, n_samples, batch
    )
    state.update(grads)
    return loss, grads
