import jax.numpy as jnp


def zscore_normalization(X, mean=None, std=None, eps=1e-10):
    """Apply z-score normalization on a given data.

    Args:
        X: numpy array, shape [batchsize, num_dims], the input dataset.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        tuple: the normalized dataset and the resulting mean and variance.
    """
    if X is None:
        return None, None, None

    if mean is None:
        mean = jnp.mean(X, axis=0)
    if std is None:
        std = jnp.std(X, axis=0)

    X_normalized = (X - mean) / (std + eps)

    return X_normalized, mean, std

def zscore_unnormalization(X_normalized, mean, std):
    """Unnormalize a given dataset.

    Args:
        X_normalized: numpy array, shape [batchsize, num_dims], the
            dataset needs to be unnormalized.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        numpy array, shape [batch_size, num_dims] the unnormalized dataset.
    """
    return X_normalized * std + mean