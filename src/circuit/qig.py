
from collections import OrderedDict
import copy

import numpy as np
import networkx as nx

from itertools import combinations


from .type import Qubits, Demand
from ..utils.graph import contract_edge, draw


QASM_FILES = [
    # "src/circuit/src/0410184_169.qasm",
    "src/circuit/src/adr4_197.qasm",
    "src/circuit/src/clip_206.qasm",
    # "src/circuit/src/cm42a_207.qasm",
    "src/circuit/src/co14_215.qasm",
    # "src/circuit/src/dc2_222.qasm",
    # "src/circuit/src/ham15_107.qasm",
    # "src/circuit/src/misex1_241.qasm",
]


class QIG:
    @staticmethod
    def from_qasm(filename: str) -> 'QIG':
        lines = open(filename, 'r').readlines()
        qubits = []
        demands = {}
        # capture all lines fitting in the format of 'cx q[0],q[1];'
        # and extract the qubits interaction
        for line in lines:
            if line.startswith('cx'):
                line = line.replace('cx q[', '').replace('];\n', '')
                line = line.replace('],q[', ' ')
                src, dst = line.split(' ')
                src, dst = int(src), int(dst)

                if src not in qubits:
                    qubits.append(src)
                if dst not in qubits:
                    qubits.append(dst)
                if (src, dst) not in demands:
                    demands[(src, dst)] = 1
                else:
                    demands[(src, dst)] += 1

        qig = QIG()
        qig.qubits = qubits
        qig.demands = demands
        qig.graph = qig.make_graph()

        return qig

    @staticmethod
    def contract_branch(graphes: list[nx.Graph], k: int, mem_limit: int, pool_size: int) -> list[nx.Graph]:
        """
        This is a recursive function to contract a list of graphes.
        For each graph, it generates k contracted graphes,
        by contracting the top k edges with highest weight.
        All contracted graphes are returned as a list,
        for further contraction.

        Finally, after n recursions, we get k^n contracted graphes.
        """
        contracted_graphes = []
        # sort graphes by least number of nodes
        # graphes = sorted(graphes, key=lambda g: len(g.nodes))
        # graphes = graphes[:pool_size]

        for graph in graphes:
            # sort edges by weight
            edges = sorted(graph.edges(data=True), key=lambda x: x[2]['demand'], reverse=True)
            # contract top k edges
            count = 0
            for edge in edges:
                p1, p2 = edge[:2]
                qubits1 = graph.nodes[p1]['qubits']
                qubits2 = graph.nodes[p2]['qubits']
                if len(qubits1) + len(qubits2) > mem_limit:
                    continue
                else:
                    count += 1
                    new_graph = copy.deepcopy(graph)
                    contract_edge(new_graph, edge[:2], inplace=True)
                    contracted_graphes.append(new_graph)

                if count >= k:
                    break
                
        return contracted_graphes


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
    
    def contract_greedy(self, mem_limit: int, inplace: bool=False) -> nx.Graph:
            """
            Group qubits in QIG to reduce total edge weight. 
            Currently not very efficient on edge selection. 
            """
            if inplace:
                graph = self.graph
            else:
                graph = copy.deepcopy(self.graph)

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

    def contract_hdware_constrained(self, mems: list[int], inplace: bool=False) -> nx.Graph:
        """
        Contract the graph with hardware constraints.
        mems is a list of memory limits for each processor.

        For example, mems can be like 8, 8, 6, 6, 4, 4, 
        then the final contracted graph will follow the constraints:
        after sorted by the number of qubits,
        the first two nodes will have at most 8 qubits,
        the next two nodes will have at most 6 qubits,
        and the last two nodes will have at most 4 qubits.
        That is, after sorting the nodes by the number of qubits,
        each node has no more qubits than the corresponding memory limit in mems.
        
        similar to contract_greedy, but now comply with the mems constraints.
        """

        def comply(mems: list[int], squbits: list[int]) -> bool:
            """
            mems: list of memory limits for each processor
            squbits: list of number of qubits on each node in the graph
            """
            assert sum(mems) >= sum(sq for sq in squbits), "Total memory limit is less than total qubits"
            sorted_qubits = sorted(squbits, reverse=True)
            for mem, qubit in zip(mems, sorted_qubits):
                if qubit > mem:
                    return False
            return True

        if inplace:
            graph = self.graph
        else:
            graph = copy.deepcopy(self.graph)

        # group most interacting qubits by merging two nodes in QIG
        while True:
            # sort edges by weight
            edges = sorted(graph.edges(data=True), key=lambda x: x[2]['demand'], reverse=True)
            # if exceed mem_num after merging, try the next edge
            candidate = None
            for edge in edges:
                p1, p2 = edge[:2]
                # qubits1 = graph.nodes[p1]['qubits']
                # qubits2 = graph.nodes[p2]['qubits']
                # if cannot comply with the memory limits, try the next edge
                squbits = { q: len(graph.nodes[q]['qubits']) for q in graph.nodes }
                squbits[p1] = squbits[p1] + squbits[p2]  # merge the two nodes
                squbits.pop(p2, None)  # remove the second node, as it will be merged
                squbits = list(squbits.values())
                if not comply(mems, squbits):
                    continue
                else:
                    candidate = edge
                    break
            # no edge can be merged
            if candidate is None:
                break
            contract_edge(graph, candidate[:2], inplace=True)


    def __contract_all_branches(self, k: int, mem_limit: int, pool_size: int) -> list[nx.Graph]:
        """
        obsolete
        Use contract_branch to contract the graph recursively.
        It stops when no more edges can be contracted, 
        i.e., the length of the returned list no longer increases.
        
        It may return up to k^n graphes,
        where n is the number of times the function is called.
        """
        self_hash = nx.weisfeiler_lehman_graph_hash(self.graph, node_attr='qubits', edge_attr='demand', iterations=10, digest_size=64)
        graphes = {self_hash: self.graph}  # use a dict to avoid duplicate graphes
        graphes = OrderedDict(graphes)  # maintain the order of insertion
        while True:
            contracted = self.contract_branch(graphes.values(), k, mem_limit, pool_size)
            condensed = []
            for graph in contracted:
                graph_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr='qubits', edge_attr='demand', iterations=10, digest_size=64)
                if graph_hash not in graphes:
                    graphes[graph_hash] = graph
                    condensed.append(graph)

            # keep the last pool_size graphes
            if len(graphes) > pool_size:
                graphes = OrderedDict(list(graphes.items())[-pool_size:])
            # if no new graphes are generated, stop
            if not condensed:
                break
        return graphes


    def draw(self, filename: str=None):
        """
        Draw the graph using matplotlib
        """
        node_labels = nx.get_node_attributes(self.graph, 'qubits')
        # convert to length of qubits
        node_labels = {node: len(qubits) for node, qubits in node_labels.items()}
        edge_labels = nx.get_edge_attributes(self.graph, 'demand')

        draw(self.graph, edge_labels=edge_labels, node_labels=node_labels, filename=filename)


class RandomQIG(QIG):
    def __init__(self, 
            qubit_num: int=10,
            demand_num: int=10,
            demand_range: tuple=(1, 11),
            ) -> None:
        super().__init__()
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
    qig = RandomQIG(64, 256, (1, 11))
    copy_qig = copy.deepcopy(qig)
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    mems = [16, 16, 12, 12, 8]

    # print(len(qig.graph.edges))
    # print(len(qig.graph.nodes))

    # qig.contract_greedy(8, inplace=True)
    qig.contract_hdware_constrained(mems, inplace=True)
    print(len(qig.graph.nodes))

    # count the total qubits on each processor
    total = 0
    # for node in qig.graph.nodes:
    #     qubit_num = len(qig.graph.nodes[node]['qubits'])
    #     total += qubit_num
    #     print(f"Node {node}: {qubit_num} qubits")
    # print(f"Total qubits: {total}")
    # for edge in qig.graph.edges(data=True):
    #     print('edge:', edge[:2], 'demand:', edge[2]['demand'])

    qig.draw('qig.png')
