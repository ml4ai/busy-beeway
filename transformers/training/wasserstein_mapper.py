import jax
import jax.numpy as jnp
import optax
from flax import nnx

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
        use_lipschitz_constraint=True,
        lipschitz_constraint_type="gp",
        wasserstein_lr=0.01,
    ):
        self.bnn = bnn
        self.gp = gp
        self.output_dim = output_dim
        self.lipschitz_f_dim = lipschitz_f_dim
        self.lipschitz_constraint_type = lipschitz_constraint_type
        assert self.lipschitz_constraint_type in ["gp", "lp"]

        self.lipschitz_f = LipschitzFunction(dim=lipschitz_f_dim)
        self.values_log = []

        self.optimiser = nnx.optimizer.Optimizer(
            self.lipschitz_f, optax.adagrad(wasserstein_lr)
        )
        self.use_lipschitz_constraint = use_lipschitz_constraint
        self.penalty_coeff = 10

    def compute_gradient_penalty(self, samples_p, samples_q, rngs: nnx.Rngs):
        rng = rngs()
        eps = jax.random.uniform(rng, (samples_p.shape[1], 1))
        X = eps * samples_p.T + (1 - eps) * samples_q.T

        if self.lipschitz_constraint_type == "gp":
            # Gulrajani2017, Improved Training of Wasserstein GANs
            return ((jnp.linalg.norm(vgrad(self.lipschitz_f, X), 2, 1) - 1) ** 2).mean()

        elif self.lipschitz_constraint_type == "lp":
            # Henning2018, On the Regularization of Wasserstein GANs
            # Eq (8) in Section 5
            return (
                (
                    jnp.clip(
                        jnp.linalg.norm(vgrad(self.lipschitz_f, X), 2, 1) - 1,
                        0.0,
                        jnp.inf,
                    )
                )
                ** 2
            ).mean()

    def wasserstein_optimisation(
        self, X, n_samples, n_steps=10, threshold=None, debug=False
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
            nnet_samples_bag = nnet_samples_bag.squeeze())

        for i in range(n_steps):
            objective = -self.calculate(nnet_samples, gp_samples)
            if debug:
                self.values_log.append(-objective.item())

            if self.use_lipschitz_constraint:
                penalty = 0.0
                for dim in range(self.output_dim):
                    penalty += self.compute_gradient_penalty(
                        nnet_samples[:, :, dim], gp_samples[:, :, dim]
                    )
                objective += self.penalty_coeff * penalty
            objective.backward()

            if threshold is not None:
                # Gradient Norm
                params = self.lipschitz_f.parameters()
                grad_norm = torch.cat([p.grad.data.flatten() for p in params]).norm()

            self.optimiser.step()
            if not self.use_lipschitz_constraint:
                for p in self.lipschitz_f.parameters():
                    p.data = torch.clamp(p, -0.1, 0.1)
            if threshold is not None and grad_norm < threshold:
                print("WARNING: Grad norm (%.3f) lower than threshold (%.3f). ", end="")
                print("Stopping optimization at step %d" % (i))
                if debug:
                    ## '-1' because the last wssr value is not recorded
                    self.values_log = self.values_log + [self.values_log[-1]] * (
                        n_steps - i - 1
                    )
                break
        for p in self.lipschitz_f.parameters():
            p.requires_grad = False


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
    ):
        self.gp = gp
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_dir = out_dir

        assert lipschitz_constraint_type in ["gp", "lp"]
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
            device=self.device,
            gpu_gp=self.gpu_gp,
            lipschitz_constraint_type=self.lipschitz_constraint_type,
        )

        # Setup logger
        self.print_info = print if logger is None else logger.info

        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    def optimize(
        self,
        num_iters,
        rngs: nnx.Rngs,
        n_samples=128,
        lr=1e-2,
        save_ckpt_every=50,
        print_every=10,
        debug=False,
    ):
        wdist_hist = []

        wasserstein_steps = self.wasserstein_steps
        prior_optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr=lr)

        # Prior loop
        for it in range(1, num_iters + 1):
            # Draw X
            X = self.data_generator.get(self.n_data)
            X = X.to(self.device)
            if not self.gpu_gp:
                X = X.to("cpu")

            # Draw functions from GP
            gp_samples = (
                self.gp.sample_functions(X.double(), n_samples)
                .detach()
                .float()
                .to(self.device)
            )
            if self.output_dim > 1:
                gp_samples = gp_samples.squeeze()

            if not self.gpu_gp:
                X = X.to(self.device)

            # Draw functions from BNN
            nnet_samples = (
                self.bnn.sample_functions(X, n_samples).float().to(self.device)
            )
            if self.output_dim > 1:
                nnet_samples = nnet_samples.squeeze()

            ## Initialisation of lipschitz_f
            self.wasserstein.lipschitz_f.apply(weights_init)

            # Optimisation of lipschitz_f
            self.wasserstein.wasserstein_optimisation(
                X,
                n_samples,
                n_steps=wasserstein_steps[1],
                threshold=self.wasserstein_threshold,
                debug=debug,
            )
            prior_optimizer.zero_grad()

            wdist = self.wasserstein.calculate(nnet_samples, gp_samples)
            wdist.backward()
            prior_optimizer.step()

            wdist_hist.append(float(wdist))
            if (it % print_every == 0) or it == 1:
                self.print_info(
                    ">>> Iteration # {:3d}: "
                    "Wasserstein Dist {:.4f}".format(it, float(wdist))
                )

            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                torch.save(self.bnn.state_dict(), path)

        # Save accumulated list of intermediate wasserstein values
        if debug:
            values = np.array(self.wasserstein.values_log).reshape(-1, 1)
            path = os.path.join(self.out_dir, "wsr_intermediate_values.log")
            np.savetxt(path, values, fmt="%.6e")
            self.print_info("Saved intermediate wasserstein values in: " + path)

        return wdist_hist
