import networkx as nx


def read_graph(path):
    """
    Read the signed graph in the networkx format.

    :param path: path for the edge list.
    :return: networkx graph
    """

    G = nx.read_edgelist(path, nodetype=int, data=(("weight", float),))

    return G