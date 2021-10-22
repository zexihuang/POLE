import argparse
from ast import literal_eval
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import scale
import numpy as np
import utils
import matrix


def postprocess_decomposition(u, s):
    """
    Postprocess the decomposed vectors and values into final embeddings.

    :param u: Eigenvectors.
    :param s: Eigenvalues.
    :return: Embeddings.
    """

    dim = len(s)

    # Weight the vectors with square root of values.
    for i in range(dim):
        u[:, i] *= np.sqrt(s[i])

    # Unify the sign of vectors for reproducible results.
    for i in range(dim):
        if u[0, i] < 0:
            u[:, i] *= -1

    # Scale the embedding matrix with mean removal and variance scaling.
    u = np.reshape(scale(u.flatten()), u.shape)

    return u


def embed(G, dim, t, signed):
    """
    Embed the graph by factorizing the dynamic similarity matrix (unsigned/signed autocovariance).

    :param G: Input graph.
    :param dim: Dimensions of embedding.
    :param t: Markov time.
    :param signed: Whether to use signed or unsigned autocovariance matrix.
    :return: Embeddings of shape (num_nodes, dim).
    """

    if signed:
        R = matrix.signed_autocovariance_matrix(G, t)
    else:
        R = matrix.unsigned_autocovariance_matrix(G, t)

    s, u = eigsh(A=R, k=dim, which='LA', maxiter=R.shape[0] * 20)
    u = postprocess_decomposition(u, s)

    return u


def parse_args():
    """
    Parse the embedding arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Embedding")

    parser.add_argument('--graph', default='graph/WoW-EP8.edges',
                        help='Input graph edgelist. Default is "graph/WoW-EP8.edges". ')

    parser.add_argument('--embedding', default='emb/WoW-EP8.embedding',
                        help='Output embedding. '
                             'Default is "emb/WoW-EP8.embedding". ')

    parser.add_argument('--dimensions', type=int, default=40,
                        help='Number of dimensions. Default is 40.')

    parser.add_argument('--markov-time', type=float, default=0.5,
                        help='Markov time in terms of power of 10. Default is 0.5 (for 10^0.5).')

    parser.add_argument('--signed', type=literal_eval, default=True,
                        help='Whether to generate signed (POLE) or unsigned (RWE) embedding. '
                             'True for signed (POLE) embedding and False for unsigned (RWE) embedding. '
                             'Default is True.')

    return parser.parse_args()


def main():
    """
    Pipeline for embedding.
    """
    args = parse_args()

    G = utils.read_graph(args.graph)
    t = 10 ** args.markov_time

    emb = embed(G, args.dimensions, t, args.signed)

    np.savetxt(args.embedding, emb, fmt='%.16f')

    print(f'Embedding done.')


if __name__ == "__main__":
    main()
