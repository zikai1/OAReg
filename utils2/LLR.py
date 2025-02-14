import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.linalg import solve
from scipy.sparse import csr_matrix
import torch

DEVICE='cuda'



def barycenter_weights(X, Y, indices, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis

    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i]. The barycenter weights sum to 1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)

    Y : array-like, shape (n_samples, n_dim)

    indices : array-like, shape (n_samples, n_dim)
            Indices of the points in Y used to compute the barycenter

    reg : float, default=1e-3
        Amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim

    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)

    Notes
    -----
    See developers note for more information.
    """
    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = check_array(Y, dtype=FLOAT_DTYPES)
    indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)


    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, assume_a="pos")
        B[i, :] = w / np.sum(w)
    return B


def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    """Computes the barycenter weighted graph of k-Neighbors for points in X

    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.

    n_neighbors : int
        Number of neighbors for each sample.

    reg : float, default=1e-3
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.

    n_jobs : int or None, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.

    See Also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='kd_tree',n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X, ind, reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))


def local_linear_reconstruction(x_canonical, x_deformed, n_neighbors=30):
    """
    Computes the locally linear reconstrunction loss between two sets of points, which measures the discrepancy
    between their linear reconstruction distances.

    Parameters
    ----------
    x_canonical : array-like, shape (n_points, n_dims)
        The canonical (reference) point set, where `n_points` is the number of points
        and `n_dims` is the number of dimensions.
    x_deformed : array-like, shape (n_points, n_dims)
        The deformed (transformed) point set, which should reconstruct the same shape as `x_canonical`.
    n_neighbors : int, optional
        The number of nearest neighbors to use for computing pairwise distances.
        Default is 30.

    Returns
    -------
    loss : float
        The LLR loss between `x_canonical` and `x_deformed`, computed as the L2 norm
        of the difference between their pairwise distances. The loss is a scalar value.
    Raises
    ------
    ValueError
        If `x_canonical` and `x_deformed` have different shapes.
    """

    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    
    X=x_canonical.cpu().numpy()
    M=barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None)
    M_data=M.data.reshape(-1,n_neighbors)
    M_data_gpu=torch.tensor(M_data).to(DEVICE)
    M_data_gpu=torch.unsqueeze(M_data_gpu,1)
    nn_ix=M.indices
    nn_ix2=nn_ix.reshape(-1,n_neighbors)
    XY_knn=x_deformed[nn_ix2]

    x_deformed_linear_composition=M_data_gpu@XY_knn
    x_deformed_linear_composition=torch.squeeze(x_deformed_linear_composition,1)

        
    x_diff=x_deformed-x_deformed_linear_composition
    
    loss=torch.sum(torch.norm(x_diff,p=2, dim=1, keepdim=True))
        
    return loss