
import copy

import numpy as np
import networkx as nx

from .type import Qubits, Demand
from ..utils.graph import contract_edge


class QIG:
    def __init__(self, ) -> None:
        self.qubits: Qubits = []
        # demands: {(src, dst): demand}
        self.demands: Demand = {}
        self.graph: nx.Graph = None

    def get_desc(self) -> 'tuple[Qubits, Demand]':
        return (self.qubits, self.demands)
    
    def make_graph(self) -> nx.Graph:
        assert self.qubits, "Qubits are not assigned"
        assert self.demands, "Demands are not assigned"

        graph = nx.Graph()
        for qubit in self.qubits:
            # assign each qubit to a node in QIG
            # record the qubit in the node
            graph.add_node(qubit, qubits=[qubit])

        for pair, demand in self.demands.items():
            graph.add_edge(*pair, demand=demand)

        return graph
    
    def contract(self, mem_limit: int, inplace: bool=False) -> nx.Graph:
            """
            Group qubits in QIG to reduce total edge weight
            Currently not very efficient on edge selection
            """
            if inplace:
                graph = self.graph
            else:
                graph = copy.deepcopy(graph)

            # group most interacting qubits by merging two nodes in QIG
            # but the total qubit on each processor should not exceed mem_num
            while True:
                # sort edges by weight
                edges = sorted(graph.edges(data=True), key=lambda x: x[2]['demand'], reverse=True)
                # if exceed mem_num after merging, try the next edge
                candidate = None
                for edge in edges:
                    p1, p2 = edge[:2]
                    qubits1 = graph.nodes[p1]['qubits']
                    qubits2 = graph.nodes[p2]['qubits']
                    if len(qubits1) + len(qubits2) > mem_limit:
                        continue
                    else:
                        candidate = edge
                        break
                # no edge can be merged
                if candidate is None:
                    break
                contract_edge(graph, edge[:2], inplace=True)

            return graph


class RandomQIG(QIG):
    def __init__(self, 
            qubit_num: int=10,
            demand_num: int=10,
            demand_range: tuple=(1, 11),
            ) -> None:
        self.qubits: Qubits = list(range(qubit_num))
        self.demands: Demand = self.demand_gen(demand_num, demand_range)

        self.graph: nx.Graph = self.make_graph()


    def demand_gen(self, demand_num: float, demand_range: tuple) -> dict:
        assert 0 <= demand_num <= len(self.qubits) * (len(self.qubits) - 1) / 2
        
        # all possible pairs of qubits
        pairs = []
        for i in range(len(self.qubits)):
            for j in range(i+1, len(self.qubits)):
                pairs.append((i, j))

        demands = {}
        selected = np.random.choice(len(pairs), demand_num, replace=False)
        for idx in selected:
            src, dst = pairs[idx]
            demands[(src, dst)] = np.random.randint(*demand_range)

        return demands



if __name__ == '__main__':
    qig = RandomQIG(64, 64*8, (1, 11))

    print(len(qig.graph.edges))
    print(len(qig.graph.nodes))

    qig.contract(8, inplace=True)


    print(qig.graph.nodes(data=True))

    # count the total qubits on each processor
    total = 0
    for node in qig.graph.nodes:
        qubit_num = len(qig.graph.nodes[node]['qubits'])
        total += qubit_num
        print(f"Node {node}: {qubit_num} qubits")
    print(f"Total qubits: {total}")


