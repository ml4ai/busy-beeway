from flax import nnx
import numpy as np
import jax.numpy as jnp
import transformers.models.parameter as parameter


class Kern(nnx.Module):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.
        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.
        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        self.name = name
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if (
                active_dims.start is not None
                and active_dims.stop is not None
                and active_dims.step is not None
            ):
                assert len(range(*active_dims)) == input_dim  # pragma: no cover
        else:
            self.active_dims = jnp.asarray(active_dims, dtype=jnp.int32)
            assert len(active_dims) == input_dim

    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        # if isinstance(self.active_dims, slice):
        X = X[:, self.active_dims]
        if X2 is not None:
            X2 = X2[:, self.active_dims]
        # I think advanced indexing does the right thing also for the second case
        # else:
        assert X.shape[1] == self.input_dim

        return X, X2

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])


class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on
        r = || x - x' ||
    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(
        self,
        input_dim,
        variance=1.0,
        lengthscales=None,
        active_dims=None,
        ARD=False,
        name=None,
    ):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        super(Stationary, self).__init__(input_dim, active_dims, name=name)
        self.variance = parameter.PositiveParam(variance)
        if ARD:
            if lengthscales is None:
                lengthscales = jnp.ones(input_dim)
            else:
                # accepts float or array:
                lengthscales = lengthscales * jnp.ones(input_dim)
            self.lengthscales = parameter.PositiveParam(lengthscales)
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.lengthscales = parameter.PositiveParam(lengthscales)
            self.ARD = False

    def square_dist(self, X, X2):
        X = X / self.lengthscales.get()
        Xs = (X**2).sum(1)

        if X2 is None:
            dist = -2 * X @ X.T
            dist += Xs.reshape(-1, 1) + Xs.reshape(1, -1)
            return dist

        X2 = X2 / self.lengthscales.get()
        X2s = (X2**2).sum(1)
        dist = -2 * X @ X2.T
        dist += Xs.reshape(-1, 1) + X2s.reshape(1, -1)
        return dist

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return (r2 + 1e-12) ** 0.5

    def Kdiag(self, X, presliced=False):
        return jnp.broadcast_to(self.variance.get(),X.shape[0])


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        res = self.variance.get() * jnp.exp(-0.5 * self.square_dist(X, X2))
        return res


class Combination(Kern):
    """
    Combine  a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).
    The names of the kernels to be combined are generated from their class
    names.
    """

    def __init__(self, kern_list, name=None):
        for k in kern_list:
            assert isinstance(k, Kern), "can only add/multiply Kern instances"

        input_dim = np.max(
            [
                (
                    k.input_dim
                    if type(k.active_dims) is slice
                    else jnp.max(k.active_dims) + 1
                )
                for k in kern_list
            ]
        )
        super(Combination, self).__init__(input_dim=input_dim, name=name)

        # add kernels to a list, flattening out instances of this class therein
        self.kern_list = []
        for k in kern_list:
            if isinstance(k, self.__class__):
                self.kern_list.extend(k.kern_list)
            else:
                self.kern_list.append(k)

    @property
    def on_separate_dimensions(self):
        """
        Checks whether the kernels in the combination act on disjoint subsets
        of dimensions. Currently, it is hard to asses whether two slice objects
        will overlap, so this will always return False.
        :return: Boolean indicator.
        """
        if jnp.any([isinstance(k.active_dims, slice) for k in self.kern_list]):
            # Be conservative in the case of a slice object
            return False
        else:
            dimlist = [k.active_dims for k in self.kern_list]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1 :]:
                    if jnp.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping


class Add(Combination):
    def K(self, X, X2=None, presliced=False):
        res = 0.0
        for k in self.kern_list:
            res += k.K(X, X2, presliced=presliced)
        return res

    def Kdiag(self, X, presliced=False):
        res = 0.0
        for k in self.kern_list:
            res += k.Kdiag(X, presliced=presliced)
        return res


class Prod(Combination):
    def K(self, X, X2=None, presliced=False):
        res = 1.0
        for k in self.kern_list:
            res *= k.K(X, X2, presliced=presliced)
        return res

    def Kdiag(self, X, presliced=False):
        res = 1.0
        for k in self.kern_list:
            res *= k.Kdiag(X, presliced=presliced)
        return res
