

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from src.circuit.qig import QIG
from src.solver import TACOORIG, TACONL, TACOL


def plot_graph(edges, filename='graph.png'):
    G = nx.Graph()
    G.add_edges_from(edges)
    layout = nx.spring_layout(G)
    nx.draw(G, layout)
    plt.savefig(filename)
    plt.close()
    


if __name__ == "__main__":
    # topo_file = 'result/topos-small.pkl'
    # topo_file = 'result/topos-small-density.pkl'
    # topo_file = 'result/topos-small-rand16.pkl'
    topo_file = 'result/topos-medium.pkl'
    # topo_file = 'result/topos-large.pkl'
    topos = pickle.load(open(topo_file, 'rb'))
    print(topos.keys())

    folder = 'result/topo/'

    qigs_names = [ "adr4", "clip", "co14", "rand16", "rand24", "rand32" ]
    clusters = [
        # (4, 4, 4, 4),
        (8, 2, 4, 8),
        (8, 2, 4, 8),
        (8, 2, 4, 10),
        (8, 2, 4, 12),
        (8, 3, 4, 8),
        (8, 3, 4, 10),
        (8, 3, 4, 12),
        (12, 2, 4, 14),
        # (8, 2, 3, 14),
        # (8, 2, 4, 16),
        (8, 4, 4, 8),
        (8, 4, 4, 10),
        (8, 4, 4, 12),
        (16, 2, 4, 20),
    ]
    solvers = [TACOORIG, TACONL, TACOL]
    sol_names = ['TACO', 'TACO-NL', 'TACO-L']



    for cluster in clusters:
        n_qpu, n_mem, n_comm, W = cluster
        for qig_name in qigs_names:
            sol_name = 'TACO-L'
            key = (cluster, qig_name, TACOL)
            if key not in topos:
                continue
            edges = topos[key]
            plot_graph(edges, f'{folder}{qig_name}-{sol_name}-{n_qpu}-{n_mem}-{n_comm}-{W}.png')
                