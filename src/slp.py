import argparse
import numpy as np
import networkx as nx
import random
from math import ceil
from sklearn.metrics import pairwise_kernels
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import utils


def link_removal(G, p, shuffle_random_state=0):
    """
    Remove links in the graph and generate the residual graph for embedding.

    :param G: Input graph.
    :param p: Ratio of removed links.
    :param shuffle_random_state: Random state for link removal.
    :return: (remaining_edges, removed_edges)
    :raise ValueError: if the given ratio of edges can't be removed while maintaining connectivity.
    """

    all_edges = list(map(tuple, np.sort(np.asarray(G.edges()))))  # Guarantee edge[0] <= edge[1]

    num_edges = len(all_edges)
    num_removed_edges = ceil(num_edges * p)
    random.Random(shuffle_random_state).shuffle(all_edges)

    # Remove edges.
    removed_edges = []
    remaining_edges = all_edges
    added_back_edges = []
    residual_G = G.copy()
    while len(removed_edges) < num_removed_edges:
        if len(remaining_edges) == 0:
            raise ValueError(f'The given ratio of edges {p} cannot be removed while maintaining connectivity. Only {len(removed_edges)} out of {num_edges} can be removed. ')
        edge = remaining_edges.pop()
        edge_data = residual_G.edges[edge]
        residual_G.remove_edge(edge[0], edge[1])
        if nx.is_connected(residual_G):  # Remove the edge if the graph is still connected.
            removed_edges.append(edge)
        else:  # Otherwise, add that edge back.
            residual_G.add_edge(edge[0], edge[1], **edge_data)
            added_back_edges.append(edge)
    remaining_edges.extend(added_back_edges)

    # Append edge weights and signs.
    remaining_edges = [(edge[0], edge[1], G.edges[edge]['weight']) for edge in remaining_edges]
    removed_edges = [(edge[0], edge[1], G.edges[edge]['weight']) for edge in removed_edges]

    return np.asarray(remaining_edges), np.asarray(removed_edges)


def prediction_ranking(emb, candidate_mask, k, reverse_ranking):
    """
    Signed link prediction utility for ranking edges (POLE).

    :param emb: Node embedding.
    :param candidate_mask: Mask of candidate edges for ranking.
    :param k: Number of top ranking pairs.
    :param reverse_ranking: Whether to reverse the ranking (for predicting negative edges).
    :return: k top ranking pairs.
    """

    similarity = pairwise_kernels(emb, metric='linear')  # Compute similarity.
    if reverse_ranking:  # Ranking from bottom to top, used when ranking negative edges.
        scores = - similarity[candidate_mask]
    else:  # Ranking from top to bottom.
        scores = similarity[candidate_mask]

    index = np.argpartition(scores, -k)[-k:]  # Find the top candidates.
    index = index[np.argsort(-scores[index])]  # Sort the top candidates
    candidate_edges = np.vstack(np.nonzero(candidate_mask)).T
    predicted_edges = candidate_edges[index]

    return predicted_edges


@ignore_warnings(category=ConvergenceWarning)
def prediction_classifier_ranking(emb, unsigned_emb, remaining_edges, candidate_mask, k):
    """
    Signed link prediction with link existence information utility for ranking edges (POLE + RWE).

    :param emb: Node embedding (POLE).
    :param unsigned_emb: Unsigned node embedding (RWE) for link existence information.
    :param remaining_edges: Edges that are still present in the graph as positive training samples.
    :param candidate_mask: Mask of candidate edges for ranking.
    :param k: Number of top ranking pairs.
    :return: k top ranking pairs.
    """

    # Compute and rescale similarities.
    signed_similarity = pairwise_kernels(emb, metric='linear')  # Compute signed similarity.
    unsigned_similarity = pairwise_kernels(unsigned_emb, metric='linear')  # Compute signed similarity.
    signed_similarity = np.triu(signed_similarity, 1)
    signed_similarity[np.triu_indices_from(signed_similarity, 1)] = scale(signed_similarity[np.triu_indices_from(signed_similarity, 1)])
    unsigned_similarity = np.triu(unsigned_similarity, 1)
    unsigned_similarity[np.triu_indices_from(unsigned_similarity, 1)] = scale(unsigned_similarity[np.triu_indices_from(unsigned_similarity, 1)])

    # Train logistic classifier to combine similarities.
    cls = LogisticRegression(solver='liblinear')
    candidate_edges = np.vstack(np.nonzero(candidate_mask)).T
    train_edges = np.vstack([remaining_edges, candidate_edges])
    train_labels = np.asarray([1] * len(remaining_edges) + [0] * len(candidate_edges))
    train_features = np.column_stack([signed_similarity[tuple(train_edges.T)], unsigned_similarity[tuple(train_edges.T)]])
    cls.fit(train_features, train_labels)

    # Compute classifier scores on the candidate edges.
    test_features = np.column_stack([signed_similarity[tuple(candidate_edges.T)], unsigned_similarity[tuple(candidate_edges.T)]])
    scores = cls.predict_proba(test_features)[:, 1]

    index = np.argpartition(scores, -k)[-k:]  # Find the top candidates.
    index = index[np.argsort(-scores[index])]  # Sort the top candidates
    predicted_edges = candidate_edges[index]

    return predicted_edges


def signed_link_prediction(emb, removed_edges, remaining_edges, k, unsigned_emb=None):
    """
    Signed link prediction.

    :param emb: Node embedding (POLE).
    :param removed_edges: Edges that have been removed from the graph.
    :param remaining_edges: Edges that are still present in the graph.
    :param k: Ratio of top ranking pairs over removed edges of the corresponding sign.
    :param unsigned_emb: Unsigned node embedding (RWE) for link existence information.
    :return: (positive_precision@k, negative_precision@k)
    """

    n = len(emb)
    removed_positive_edges = removed_edges[:, :2][removed_edges[:, 2] > 0].astype(int)
    removed_negative_edges = removed_edges[:, :2][removed_edges[:, 2] <= 0].astype(int)
    remaining_positive_edges = remaining_edges[:, :2][remaining_edges[:, 2] > 0].astype(int)
    remaining_negative_edges = remaining_edges[:, :2][remaining_edges[:, 2] <= 0].astype(int)

    # Compute the actual k in terms of number of edges.
    k_positive = int(np.ceil(len(removed_positive_edges) * k))
    k_negative = int(np.ceil(len(removed_negative_edges) * k))

    # Mask the candidate edges for ranking.
    candidate_mask = np.triu(np.ones(shape=(n, n), dtype=bool), 1)
    candidate_mask[tuple(remaining_positive_edges.T)] = False  # Excluding existing positive edges.
    candidate_mask[tuple(remaining_negative_edges.T)] = False  # Excluding existing negative edges.

    # Predict the top ranking pairs.
    if unsigned_emb is None:  # Signed link prediction (POLE).
        predicted_positives = prediction_ranking(emb, candidate_mask, k=k_positive, reverse_ranking=False)
        predicted_negatives = prediction_ranking(emb, candidate_mask, k=k_negative, reverse_ranking=True)
    else:
        predicted_positives = prediction_classifier_ranking(emb, unsigned_emb, remaining_positive_edges, candidate_mask, k=k_positive)
        predicted_negatives = prediction_classifier_ranking(emb, unsigned_emb, remaining_negative_edges, candidate_mask, k=k_negative)

    # Compute the precision scores.
    positive_tps = len(set(map(tuple, removed_positive_edges)).intersection(set(map(tuple, predicted_positives))))
    negative_tps = len(set(map(tuple, removed_negative_edges)).intersection(set(map(tuple, predicted_negatives))))
    positive_precision = positive_tps / k_positive
    negative_precision = negative_tps / k_negative

    return positive_precision, negative_precision


def parse_args():
    """
    Parse the evaluating arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Signed Link Prediction")

    parser.add_argument('--mode', default='prepare',
                        help='Mode of signed link prediction evaluation. '
                             'Options: "preparation" (remove edges in the graph and generate residual graph for embedding), '
                             '"slp" (signed link prediction), '
                             '"slp-rwe" (signed link prediction with link existence information from RWE embedding).' 
                             'Default is "preparation". ')

    # Preparation parameters.
    parser.add_argument('--graph', default='graph/WoW-EP8.edges',
                        help='Input graph edgelist for link removal in "preparation" mode. '
                             'Default is "graph/WoW-EP8.edges". ')

    parser.add_argument('--remove-ratio', type=float, default=0.2,
                        help='Ratio of removed edges in link removal in "preparation" mode. Default is 0.2. ')

    parser.add_argument('--remaining-edges', default='graph/WoW-EP8.remaining-edges',
                        help='Remaining edges after link removal. '
                             'Serve as output in "preparation" mode and input in "slp" and "slp-rwe" modes. '
                             'Default is "graph/WoW-EP8.remaining-edges". ')

    parser.add_argument('--removed-edges', default='graph/WoW-EP8.removed-edges',
                        help='Removed edges in link removal. '
                             'Serve as output in "preparation" mode and input in "slp" and "slp-rwe" modes. '
                             'Default is "graph/WoW-EP8.removed-edges". ')

    # SLP and SLP-RWE parameters.
    parser.add_argument('--embedding', default='emb/WoW-EP8.residual-embedding',
                        help='Input signed embedding (POLE) in "slp" and "slp-rwe" modes.'
                             'Default is "emb/WoW-EP8.residual-embedding". ')

    parser.add_argument('--unsigned-embedding', default='emb/WoW-EP8.residual-unsigned-embedding',
                        help='Input unsigned embedding (RWE) for link existence information in "slp-rwe" mode.'
                             'Default is "emb/WoW-EP8.residual-unsigned-embedding". ')

    parser.add_argument('--k', type=float, default=1.0,
                        help='Ratio of top ranking pairs over removed edges in "slp" and "slp-rwe" modes. '
                             'Default is 1.0. ')

    return parser.parse_args()


def main():
    """
    Pipeline for signed link prediction.
    """
    args = parse_args()

    if args.mode == 'preparation':  # Remove a proportion of edges in the original graph for embedding.
        G = utils.read_graph(args.graph)
        remaining_edges, removed_edges = link_removal(G, args.remove_ratio)
        fmt = ['%d', '%d', '%f']
        np.savetxt(args.remaining_edges, remaining_edges, fmt=fmt)
        np.savetxt(args.removed_edges, removed_edges, fmt=fmt)
    elif args.mode == 'slp':  # Signed link prediction with POLE embedding.
        emb = np.loadtxt(args.embedding)
        removed_edges = np.loadtxt(args.removed_edges)
        remaining_edges = np.loadtxt(args.remaining_edges)
        positive_precision, negative_precision = signed_link_prediction(emb, removed_edges, remaining_edges, args.k)
        print(f'SLP: positive precision@{args.k:.0%} = {positive_precision:.4f}; negative precision@{args.k:.0%} = {negative_precision:.4f}')
    elif args.mode == 'slp-rwe':
        emb = np.loadtxt(args.embedding)
        unsigned_emb = np.loadtxt(args.unsigned_embedding)
        removed_edges = np.loadtxt(args.removed_edges)
        remaining_edges = np.loadtxt(args.remaining_edges)
        positive_precision, negative_precision = signed_link_prediction(emb, removed_edges, remaining_edges, args.k, unsigned_emb)
        print(f'SLP-RWE: positive precision@{args.k:.0%} = {positive_precision:.4f}; negative precision@{args.k:.0%} = {negative_precision:.4f}')
    else:
        raise NotImplementedError(f'Signed link prediction mode {args.mode} not implemented. ')


if __name__ == "__main__":
    main()