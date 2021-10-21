import networkx as nx
import numpy as np
from scipy.sparse.linalg import expm


def signed_adjacency_matrix(G):
    """
    Adjacency matrix for the signed graph with positive/negative edge weights.
    A.

    :param G: Input graph.
    :return: Signed adjacency matrix.
    """
    return nx.adjacency_matrix(G, sorted(G.nodes), weight='weight').toarray()


def unsigned_adjacency_matrix(G):
    """
    Adjacency matrix for the unsigned graph with absolute edge weights.
    A_abs.

    :param G: Input graph.
    :return: Unsigned adjacency matrix.
    """
    return np.abs(signed_adjacency_matrix(G))


def unsigned_degree_vector(G):
    """
    Degree vector for the unsigned graph.
    d.

    :param G: Input graph.
    :return: Unsigned degree vector.
    """
    A_abs = unsigned_adjacency_matrix(G)
    return A_abs.sum(axis=1)


def unsigned_random_walk_stationary_distribution_vector(G):
    """
    Stationary distribution vector for unsigned random-walk dynamics.
    pi = d/vol(G).

    :param G: Input graph.
    :return: Unsigned random-walk stationary distribution vector.
    """
    d_abs = unsigned_degree_vector(G)
    return d_abs/np.sum(d_abs)


def unsigned_random_walk_stationary_distribution_matrix(G):
    """
    Stationary distribution matrix for unsigned random-walk dynamics.
    Pi = diag(pi).

    :param G: Input graph.
    :return: Unsigned random-walk stationary distribution matrix.
    """
    return np.diag(unsigned_random_walk_stationary_distribution_vector(G))


def signed_random_walk_laplacian_matrix(G):
    """
    Random-walk Laplacian matrix for the signed graph.
    L_rw = I - D^-1 A.

    :param G: Input graph.
    :return: Signed random-walk Laplacian matrix.
    """
    D_1 = np.diag(1/unsigned_degree_vector(G))
    A = signed_adjacency_matrix(G)
    return np.identity(G.number_of_nodes()) - D_1 @ A


def unsigned_random_walk_laplacian_matrix(G):
    """
    Random-walk Laplacian matrix for the unsigned graph.
    L_rw_abs = I - D^-1 A_abs.

    :param G: Input graph.
    :return: Unsigned random-walk Laplacian matrix.
    """
    D_1 = np.diag(1/unsigned_degree_vector(G))
    A_abs = unsigned_adjacency_matrix(G)
    return np.identity(G.number_of_nodes()) - D_1 @ A_abs


def transition_matrix(L, t):
    """
    Transition matrix based on the Laplacian matrix and Markov time.
    P(t) = exp(-Lt).

    :param L: Laplacian matrix.
    :param t: Markov time.
    :return: Transition matrix.
    """
    return expm(- L * t)


def dynamic_similarity_matrix(M_t, W):
    """
    Dynamic similarity matrix.
    R(t) = M(t)^T W M(t).

    :param M_t: Transition matrix.
    :param W: Weight matrix.
    :return: Dynamic similarity matrix.
    """
    return M_t.T @ W @ M_t


def signed_autocovariance_matrix(G, t):
    """
    Signed autocovariance matrix based on signed transition matrix and unsigned stationary distributions.
    R = M(t)^T (Pi - pi pi^T) M(t).

    :param G: Input graph.
    :param t: Markov time.
    :return: Signed autocovariance similarity matrix.
    """

    pi = unsigned_random_walk_stationary_distribution_vector(G)
    Pi = unsigned_random_walk_stationary_distribution_matrix(G)
    W = Pi - np.outer(pi, pi)

    L_rw = signed_random_walk_laplacian_matrix(G)
    M_t = transition_matrix(L_rw, t)

    return dynamic_similarity_matrix(M_t, W)


def unsigned_autocovariance_matrix(G, t):
    """
    Signed autocovariance matrix based on unsigned transition matrix and unsigned stationary distributions.
    R(t)_abs = M(t)_abs^T (Pi - pi pi^T) M(t)_abs.

    :param G: Input graph.
    :param t: Markov time.
    :return: Unsigned autocovariance similarity matrix.
    """

    pi = unsigned_random_walk_stationary_distribution_vector(G)
    Pi = unsigned_random_walk_stationary_distribution_matrix(G)
    W = Pi - np.outer(pi, pi)

    L_rw = unsigned_random_walk_laplacian_matrix(G)
    M_t = transition_matrix(L_rw, t)

    return dynamic_similarity_matrix(M_t, W)