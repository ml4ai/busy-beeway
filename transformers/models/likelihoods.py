import jax.numpy as jnp
from flax import nnx

import transformers.models.quadrature as quadrature
import transformers.models.densities as densities
import transformers.models.parameter as parameter

class Likelihood(nnx.Module):
    def __init__(self, name=None):
        self.name = name
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w /= float(jnp.pi**0.5)
        gh_w = gh_w.reshape(-1, 1)
        shape = Fmu.shape
        Fmu = Fmu.reshape(-1, 1)
        Fvar = Fvar.reshape(-1, 1)
        X = gh_x[None, :] * (2.0 * Fvar) ** 0.5 + Fmu

        # here's the quadrature for the mean
        E_y = (self.conditional_mean(X) @ gh_w).reshape(shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X) + (self.conditional_mean(X)) ** 2
        V_y = (integrand @ gh_w).reshape(shape) - E_y**2

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.
        i.e. if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive density
           \int p(y=Y|f)q(f) df
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = quadrature.hermgauss(
            self.num_gauss_hermite_points, dtype=Fmu.dtype
        )

        gh_w = gh_w.reshape(-1, 1) / float(jnp.sqrt(jnp.pi))
        shape = Fmu.shape
        Fmu, Fvar, Y = [e.reshape(-1, 1) for e in (Fmu, Fvar, Y)]
        X = gh_x * (2.0 * Fvar) ** 0.5 + Fmu
        Y = jnp.broadcast_to(Y,(Y.shape[0], self.num_gauss_hermite_points))  # broadcast Y to match X
        logp = self.logp(X, Y)
        return (logp.exp() @ gh_w).reshape(shape)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes
           \int (\log p(y|f)) q(f) df.
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """

        gh_x, gh_w = quadrature.hermgauss(
            self.num_gauss_hermite_points, dtype=Fmu.dtype
        )
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1) / float(jnp.pi) ** 0.5
        shape = Fmu.shape
        Fmu, Fvar, Y = [e.reshape(-1, 1) for e in (Fmu, Fvar, Y)]
        X = gh_x * (2.0 * Fvar) ** 0.5 + Fmu
        Y = jnp.broadcast_to(Y,(Y.shape[0], self.num_gauss_hermite_points))  # broadcast Y to match X
        logp = self.logp(X, Y)
        return (logp @ gh_w).reshape(shape)

    def _check_targets(self, Y_np):  # pylint: disable=R0201
        """
        Check that the Y values are valid for the likelihood.
        Y_np is a np array.
        The base class check is that the array has two dimensions
        and consists only of floats. The float requirement is so that AutoFlow
        can work with Model.predict_density.
        """
        if not Y_np.ndim == 2:
            raise ValueError("targets must be shape N x D")
        # if np.array(list(Y_np)).dtype != settings.np_float:
        #    raise ValueError('use {}, even for discrete variables'.format(settings.np_float))


class Gaussian(Likelihood):
    def __init__(self, dtype=jnp.float32):
        Likelihood.__init__(self)
        self.variance = parameter.PositiveParam(
            jnp.asarray([1.0], dtype=dtype), dtype=dtype
        )

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance.get())

    def conditional_mean(self, F):
        return F

    def conditional_variance(self, F):
        return jnp.broadcast_to(self.variance.get(),F.shape)

    def predict_mean_and_var(self, Fmu, Fvar):
        return Fmu, Fvar + self.variance.get()

    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.variance.get())

    def variational_expectations(self, Fmu, Fvar, Y):
        return (
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(self.variance.get())
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance.get()
        )
