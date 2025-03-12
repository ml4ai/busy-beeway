import abc

import jax
import jax.numpy as jnp
import numpy as np
import transformers.models.densities as densities
import transformers.models.likelihoods as likelihoods
import transformers.models.mean_functions as mean_functions
import transformers.models.parameter as parameter
from flax import nnx


class GPModel(nnx.Module):
    """
    A base class for Gaussian process models, that is, those of the form
       \begin{align}
       \theta & \sim p(\theta) \\
       f       & \sim \mathcal{GP}(m(x), k(x, x'; \theta)) \\
       f_i       & = f(x_i) \\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.
    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.
    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.
    For handling another data (Xnew, Ynew), set the new value to
    self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(
        self,
        X,
        Y,
        kern,
        likelihood,
        mean_function,
        name=None,
        jitter_level=1e-6,
        rngs=nnx.Rngs(0, params=1, dropout=2),
    ):
        self.name = name
        self.mean_function = mean_function or mean_functions.Zero()
        self.kern = kern
        self.likelihood = likelihood
        self.jitter_level = jitter_level
        self.rngs = rngs

        if isinstance(X, np.ndarray):
            # X is a data matrix; each row represents one instance
            X = jnp.asarray(X)
        if isinstance(Y, np.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = jnp.asarray(Y)
        self.X, self.Y = X, Y

    # @abc.abstractmethod
    # def compute_log_prior(self):
    #     """Compute the log prior of the model."""
    #     pass

    @abc.abstractmethod
    def compute_log_likelihood(self, X=None, Y=None):
        """Compute the log likelihood of the model."""
        pass

    def objective(self, X=None, Y=None):
        pos_objective = self.compute_log_likelihood(X, Y)
        params = nnx.variables(self, nnx.Param)
        for param in jax.tree.flatten(params)[0]:
            if isinstance(param, parameter.ParamWithPrior):
                pos_objective = pos_objective + param.get_prior()
        return -pos_objective

    def __call__(self, X=None, Y=None):
        return self.objective(X, Y)

    @abc.abstractmethod
    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        raise NotImplementedError

    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.predict_f(Xnew, full_cov=True)

    def sample_functions(self, X, num_samples):
        """
        Produce samples from the prior latent functions at the points X.
        """
        X = X.reshape((-1, self.kern.input_dim))
        mu = self.mean_function(X)
        prior_params = []
        params = nnx.variables(self.kern, nnx.Param)
        for param in jax.tree.flatten(params)[0]:
            if param.prior is not None:
                prior_params.append(param)
        if len(prior_params) == 0:
            var = self.kern.K(X)
            jitter = jnp.eye(mu.shape[0], dtype=mu.dtype) * self.jitter_level
            samples = []
            for i in range(self.num_latent):
                L = jnp.linalg.cholesky(var + jitter, upper=False)
                rng = self.rngs()
                V = jax.random.normal(rng, (L.shape[0], num_samples), dtype=L.dtype)
                samples.append(mu + L @ V)
            return jnp.stack(samples, axis=0).transpose(1, 2, 0)
        else:
            samples = []
            for sample_idx in range(num_samples):
                for param in prior_params:
                    jax.lax.stop_gradient(param.copy(param.prior.sample()))
                var = self.kern.K(X)
                jitter = jnp.eye(mu.shape[0], dtype=mu.dtype) * self.jitter_level
                s = []
                for i in range(self.num_latent):
                    multiplier = 1
                    while True:
                        try:
                            L = jnp.linalg.cholesky(
                                var + multiplier * jitter, upper=False
                            )
                            break
                        except RuntimeError as err:
                            multiplier *= 2
                    rng = self.rngs()
                    V = jax.random.normal(rng, (L.shape[0], 1), dtype=L.dtype)
                    s.append(mu + L @ V)
                samples.append(jnp.stack(s, axis=0).transpose(1, 2, 0))
            return jnp.concatenate(samples, axis=1)

    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.predict_f(Xnew, full_cov=True)
        jitter = jnp.eye(mu.shape[0], dtype=mu.dtype) * self.jitter_level
        samples = []
        for i in range(self.num_latent):  # TV-Todo: batch??
            L = jnp.linalg.cholesky(var[:, :, i] + jitter, upper=False)
            rng = self.rngs()
            V = jax.random.normal(rng, (L.shape[0], num_samples), dtype=L.dtype)
            samples.append(mu[:, i : i + 1] + L @ V)
        return jnp.stack(samples, axis=0)  # TV-Todo: transpose?

    def predict_y(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.predict_f(Xnew, full_cov)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.predict_f(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    def _repr_html_(self):
        s = "Model {}<ul>".format(type(self).__name__)
        for n, c in self.named_children():
            s += "<li>{}: {}</li>".format(n, type(c).__name__)
        s += "</ul><table><tr><th>Parameter</th><th>Value</th><th>Prior</th><th>ParamType</th></tr><tr><td>"
        s += "</td></tr><tr><td>".join(
            [
                "</td><td>".join(
                    (n, str(p.get().data.cpu().numpy()), str(p.prior), type(p).__name__)
                )
                for n, p in self.named_parameters()
            ]
        )
        s += "</td></tr></table>"
        return s


class GPR(GPModel):
    """Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently. The log
    likelihood i this models is sometimes referred to as the 'marginal
    log likelihood', and is given by

    \log p(\mathbf y \,|\, \mathbf f) = \mathcal N\left(\mathbf y\,|\, 0, \mathbf K + \sigma_n \mathbf I\right)
    """

    def __init__(
        self,
        X,
        Y,
        kern,
        mean_function=None,
        rngs=nnx.Rngs(0, params=1, dropout=2),
        **kwargs
    ):
        """Initialization

        Args:
            X is a data matrix, size N x D
            Y is a data matrix, size N x R
        """
        likelihood = likelihoods.Gaussian(dtype=X.dtype)
        super(GPR, self).__init__(
            X, Y, kern, likelihood, mean_function, rngs=rngs, **kwargs
        )
        self.num_latent = Y.shape[1]

    def compute_log_likelihood(self, X=None, Y=None):
        """Construct function to compute the likelihood."""
        # assert X is None and Y is None, "{} does not support minibatch mode".format(str(type(self)))
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        K = self.kern.K(X)
        jitter = self.jitter_level

        if self.likelihood.variance.get() != 0.0:
            K = K + jnp.eye(X.shape[0], dtype=X.dtype) * self.likelihood.variance.get()
        else:
            K = K + jnp.eye(X.shape[0], dtype=X.dtype) * jitter

        multiplier = 1
        while True:
            try:
                L = jnp.linalg.cholesky(K + multiplier * jitter, upper=False)
                break
            except RuntimeError as err:
                multiplier *= 2.0
                if float(multiplier) == float("inf"):
                    raise RuntimeError("increase to inf jitter")
        m = self.mean_function(X)

        if Y.shape[1] > 1:
            results = []
            for i in range(int(Y.shape[1])):
                results.append(
                    densities.multivariate_normal(
                        jnp.float32(Y[:, i]), jnp.float32(m), jnp.float32(L)
                    )
                )
            return jnp.stack(results)
        else:
            return densities.multivariate_normal(Y, m, L)

    def predict_f(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes
            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        Kx = self.kern.K(self.X, Xnew)
        K = (
            self.kern.K(self.X)
            + jnp.eye(self.X.shape[0], dtype=self.X.dtype)
            * self.likelihood.variance.get()
        )
        L = jnp.linalg.cholesky(K, upper=False)

        A = jnp.linalg.solve(
            L, Kx
        )  # could use triangular solve, note gesv has B first, then A in AX=B
        V = jnp.linalg.solve(
            L, self.Y - self.mean_function(self.X)
        )  # could use triangular solve

        fmean = A.T @ V + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - A.T @ A
            fvar = jnp.broadcast_to(
                jnp.expand_dims(fvar, 2),
                (fvar.shape[0], fvar.shape[1], self.Y.shape[1]),
            )
        else:
            fvar = self.kern.Kdiag(Xnew) - (A**2).sum(0)
            fvar = fvar.reshape(-1, 1)
            fvar = jnp.broadcast_to(fvar, (fvar.shape[0], self.Y.shape[1]))

        return fmean, fvar
