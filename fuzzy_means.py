import numpy as np


def cmeans(data, c, m=2.3, epsilon=1e-3, maxiter=int(1e3), seed=42):
    data = data.iloc[:, list(range(20))].transpose()
    np.random.seed(seed=seed)
    n = data.shape[1]
    u0 = np.random.rand(c, n)
    u0 = normalize_columns(u0)
    init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Main cmeans loop
    for _ in range(maxiter):
        u2 = u.copy()
        [cntr, u, _, d] = _cmeans0(data, u2, c, m, 'euclidean')

        # Stopping rule
        if np.linalg.norm(u - u2) < epsilon:
            break

    return cntr, u, u0, d


def _cmeans0(data, u_old, c, m, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.
    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.
    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old**m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d**2).sum()

    u = normalize_power_columns(d, -2. / (m - 1))

    return cntr, u, jm, d


def normalize_columns(columns):
    """
    Normalize columns of matrix.
    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns
    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns / np.sum(columns, axis=0, keepdims=1)

    return normalized_columns


def normalize_power_columns(x, exponent):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.
    Parameters
    ----------
    x : 2d array (M x N)
        Matrix with columns
    n : float
        Exponent
    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**n) but safe
    """

    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    # values in range [0, 1]
    x = x / np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x**exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x**exponent

    result = normalize_columns(x)

    return result
