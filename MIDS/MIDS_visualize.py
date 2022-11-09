import itertools
import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make_edges(n):
    first = '0' * n

    edges = set()

    def make_edges_r(orig):
        for i, s in enumerate(orig):
            new = orig[:i] + ('1' if s == '0' else '0') + orig[i+1:]
            if not ((orig, new) in edges or (new, orig) in edges):
                edges.add((orig, new))
                make_edges_r(new)

    make_edges_r(first)

    return edges


def main():
    num_nodes = 6

    G = nx.connected_watts_strogatz_graph(num_nodes, max(int(math.sqrt(num_nodes)), 2), 0.5)
    A = nx.to_numpy_array(G)
    I = np.eye(num_nodes)

    invalid = []
    every = []
    for comb in itertools.product('01', repeat=num_nodes):
        comb = ''.join(comb)
        every.append(comb)
        d = np.array([int(s) for s in comb])
        J = (A + I) @ d
        if not np.all(J >= 1):
            invalid.append(comb)

    H = nx.Graph()
    H.add_nodes_from(every)
    H.add_edges_from(make_edges(num_nodes))
    H.remove_nodes_from(invalid)
    planar, _ = nx.check_planarity(H)
    print(f"The graph is {'' if planar else 'NOT '}planar.")

    if planar:
        nx.draw(H, pos=nx.planar_layout(H), with_labels=True, node_size=num_nodes * 300)
    else:
        nx.draw(H, pos=nx.kamada_kawai_layout(H), with_labels=True, node_size=num_nodes * 300)
    plt.show()




if __name__ == '__main__':
    main()
