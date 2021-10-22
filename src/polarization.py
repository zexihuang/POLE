import argparse
import numpy as np
from scipy.stats import pearsonr
from ast import literal_eval
import utils
import matrix


def compute_polarization(G, t, node_level):
    """
    Compute node-level or graph-level polarization scores based on correlation of signed/unsigned random-walk transitions.

    :param G: Input graph.
    :param t: Markov time.
    :param node_level: Whether to return node-level or graph-level polarization.
    :return: node-level polarization scores if node_level is True else graph-level polarization score.
    """

    M = matrix.transition_matrix(matrix.signed_random_walk_laplacian_matrix(G), t)
    M_abs = matrix.transition_matrix(matrix.unsigned_random_walk_laplacian_matrix(G), t)

    node_scores = np.asarray([pearsonr(signed, unsigned)[0] for signed, unsigned in zip(M.T, M_abs.T)])
    node_scores = np.nan_to_num(node_scores, nan=0.0)  # Replace Nan with 0.0.

    if node_level:
        return node_scores.reshape(-1, 1)
    else:
        return node_scores.mean()


def parse_args():
    """
    Parse the polarization arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Polarization")

    parser.add_argument('--node-level', type=literal_eval, default=True,
                        help='Whether to compute node-level or graph-level polarization.'
                             'True for node-level polarization and False for graph-level polarization. '
                             'Default is True. ')

    # Preparation parameters.
    parser.add_argument('--graph', default='graph/Congress.edges',
                        help='Input graph edgelist for computing polarization. '
                             'Default is "graph/Congress.edges". ')

    parser.add_argument('--markov-time', type=float, default=1.0,
                        help='Markov time in terms of power of 10. Default is 1.0 (for 10^1.0).')

    parser.add_argument('--node-polarization', default='pol/Congress.node-polarization',
                        help='Output node-level polarization scores.  '
                             'Default is "pol/Congress.node-polarization". ')

    return parser.parse_args()


def main():
    """
    Pipeline for computing polarization scores.
    """
    args = parse_args()

    G = utils.read_graph(args.graph)
    t = 10 ** args.markov_time

    score = compute_polarization(G, t, args.node_level)
    if args.node_level:
        np.savetxt(args.node_polarization, score, fmt='%.4f')
        print(f'Node-level polarization done.')
    else:
        print(f'Graph-level polarization: {score:.4f}')


if __name__ == "__main__":
    main()
