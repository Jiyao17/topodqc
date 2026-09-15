
from collections import Counter, OrderedDict
import copy
import heapq

import numpy as np
import networkx as nx

from itertools import combinations


from .type import Qubits, Demand
from ..utils.graph import contract_edge, draw


QASM_FILES = {
    ('QFT', 4): 'src/circuit/src/qft_4.qasm',
    ('QFT', 8): 'src/circuit/src/qft_8.qasm',
    ('QFT', 16): 'src/circuit/src/qft_16.qasm',
    ('QFT', 24): 'src/circuit/src/qft_24.qasm',
    ('QFT', 32): 'src/circuit/src/qft_32.qasm',
    ('QFT', 48): 'src/circuit/src/qft_48.qasm',
    ('QFT', 64): 'src/circuit/src/qft_64.qasm',
    ('QFT', 96): 'src/circuit/src/qft_96.qasm',
    ('QFT', 128): 'src/circuit/src/qft_128.qasm',
    ('QFT', 192): 'src/circuit/src/qft_192.qasm',
    ('QFT', 256): 'src/circuit/src/qft_256.qasm',
    ('QFT', 384): 'src/circuit/src/qft_384.qasm',
    ('QFT', 512): 'src/circuit/src/qft_512.qasm',
    ('QFT', 768): 'src/circuit/src/qft_768.qasm',
    ('QFT', 1024): 'src/circuit/src/qft_1024.qasm',
    ('Grover', 4): 'src/circuit/src/grover_4.qasm',
    ('Grover', 8): 'src/circuit/src/grover_8.qasm',
    ('Grover', 16): 'src/circuit/src/grover_16.qasm',
    ('Grover', 24): 'src/circuit/src/grover_24.qasm',
    ('Grover', 32): 'src/circuit/src/grover_32.qasm',
    ('Grover', 48): 'src/circuit/src/grover_48.qasm',
    ('Grover', 64): 'src/circuit/src/grover_64.qasm',
    ('Grover', 96): 'src/circuit/src/grover_96.qasm',
    ('Grover', 128): 'src/circuit/src/grover_128.qasm',
    ('Grover', 192): 'src/circuit/src/grover_192.qasm',
    ('Grover', 256): 'src/circuit/src/grover_256.qasm',
    ('Grover', 384): 'src/circuit/src/grover_384.qasm',
    ('Grover', 512): 'src/circuit/src/grover_512.qasm',
    ('Grover', 768): 'src/circuit/src/grover_768.qasm',
    ('Grover', 1024): 'src/circuit/src/grover_1024.qasm',
    ('QAOA', 4): 'src/circuit/src/qaoa_4.qasm',
    ('QAOA', 8): 'src/circuit/src/qaoa_8.qasm',
    ('QAOA', 16): 'src/circuit/src/qaoa_16.qasm',
    ('QAOA', 24): 'src/circuit/src/qaoa_24.qasm',
    ('QAOA', 32): 'src/circuit/src/qaoa_32.qasm',
    ('QAOA', 48): 'src/circuit/src/qaoa_48.qasm',
    ('QAOA', 64): 'src/circuit/src/qaoa_64.qasm',
    ('QAOA', 72): 'src/circuit/src/qaoa_72.qasm',
    ('QAOA', 96): 'src/circuit/src/qaoa_96.qasm',
    ('QAOA', 128): 'src/circuit/src/qaoa_128.qasm',
    ('QAOA', 256): 'src/circuit/src/qaoa_256.qasm',
    ('QAOA', 512): 'src/circuit/src/qaoa_512.qasm',
    ('MCMT', 4): 'src/circuit/src/mcmt_2c_2t.qasm',
    ('MCMT', 8): 'src/circuit/src/mcmt_4c_4t.qasm',
    ('MCMT', 16): 'src/circuit/src/mcmt_8c_8t.qasm',
    ('MCMT', 24): 'src/circuit/src/mcmt_12c_12t.qasm',
    ('MCMT', 32): 'src/circuit/src/mcmt_16c_16t.qasm',
    ('MCMT', 64): 'src/circuit/src/mcmt_32c_32t.qasm',
    ('MCMT', 96): 'src/circuit/src/mcmt_48c_48t.qasm',
    ('MCMT', 128): 'src/circuit/src/mcmt_64c_64t.qasm',
    ('MCMT', 192): 'src/circuit/src/mcmt_96c_96t.qasm',
    ('MCMT', 256): 'src/circuit/src/mcmt_128c_128t.qasm',
    ('MCMT', 384): 'src/circuit/src/mcmt_192c_192t.qasm',
    ('MCMT', 512): 'src/circuit/src/mcmt_256c_256t.qasm',
    ('MCMT', 768): 'src/circuit/src/mcmt_384c_384t.qasm',
    ('MCMT', 1024): 'src/circuit/src/mcmt_512c_512t.qasm',
    # ('MCMT', 512): 'src/circuit/src/mcmt_256c_256t.qasm',
}



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

        def comply(capacities: list[int], size_counts: Counter) -> bool:
            """
            Greedily place the largest superqubits into the QPU with the
            largest remaining capacity. This is equivalent to repeatedly
            sorting both lists in the original implementation, but uses one
            group-size sort and a max heap for the remaining capacities.
            """
            remaining = [-capacity for capacity in capacities]
            heapq.heapify(remaining)
            for size, count in sorted(size_counts.items(), reverse=True):
                for _ in range(count):
                    if not remaining:
                        return False
                    capacity = -heapq.heappop(remaining)
                    if size > capacity:
                        return False
                    capacity -= size
                    if capacity:
                        heapq.heappush(remaining, -capacity)
            return True

        if inplace:
            graph = self.graph
        else:
            graph = copy.deepcopy(self.graph)

        sizes = {
            node: len(graph.nodes[node]['qubits'])
            for node in graph.nodes
        }
        assert sum(mems) >= sum(sizes.values()), \
            "Total memory limit is less than total qubits"
        size_counts = Counter(sizes.values())

        # group most interacting qubits by merging two nodes in QIG
        while True:
            # A linear-time heap construction replaces sorting all edges. The
            # enumeration index preserves the original implementation's stable
            # tie order for equal-demand edges.
            edge_heap = [
                (-data['demand'], order, (src, dst))
                for order, (src, dst, data) in enumerate(graph.edges(data=True))
            ]
            heapq.heapify(edge_heap)
            # if exceed mem_num after merging, try the next edge
            candidate = None
            # Within one contraction iteration, any two edges whose endpoint
            # groups have the same sizes produce the same feasibility problem.
            feasibility_cache = {}
            while edge_heap:
                _, _, (p1, p2) = heapq.heappop(edge_heap)

                size1, size2 = sizes[p1], sizes[p2]
                cache_key = (min(size1, size2), max(size1, size2))
                feasible = feasibility_cache.get(cache_key)
                if feasible is None:
                    candidate_counts = size_counts.copy()
                    candidate_counts[size1] -= 1
                    candidate_counts[size2] -= 1
                    candidate_counts[size1 + size2] += 1
                    candidate_counts += Counter()  # discard zero counts
                    feasible = comply(mems, candidate_counts)
                    feasibility_cache[cache_key] = feasible

                if not feasible:
                    continue
                candidate = (p1, p2)
                break
            # no edge can be merged
            if candidate is None:
                break

            p1, p2 = candidate
            size1, size2 = sizes[p1], sizes[p2]
            contract_edge(graph, (p1, p2), inplace=True)

            size_counts[size1] -= 1
            size_counts[size2] -= 1
            size_counts[size1 + size2] += 1
            size_counts += Counter()
            sizes[p1] = size1 + size2
            del sizes[p2]

        return graph

    def partition_kernighan_lin(
            self,
            mems: list[int],
            inplace: bool = False,
            seed: int = 42,
            max_iter: int = 10,
            ) -> nx.Graph:
        """Partition an uncontracted QIG with recursive Kernighan--Lin.

        This provides a conventional graph-partitioning baseline for the
        proposed greedy edge-contraction preprocessor.  Each edge is weighted
        by its CNOT demand, so Kernighan--Lin minimizes the demand crossing
        partitions.  Partition sizes are selected subject to the QPU memory
        capacities in ``mems``; the resulting partitions are then represented
        as superqubits and can be passed to the existing TACO solvers without
        any solver changes.

        The method is intended as an alternative first preprocessing step and
        therefore expects the original QIG (one logical qubit per graph node).
        """
        if not mems:
            raise ValueError("At least one QPU memory capacity is required.")
        if any(not isinstance(mem, (int, np.integer)) or mem <= 0 for mem in mems):
            raise ValueError("All QPU memory capacities must be positive integers.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        graph = self.graph
        if graph is None:
            raise ValueError("QIG graph is not initialized.")
        if graph.number_of_nodes() == 0:
            partitioned = copy.deepcopy(graph)
            if inplace:
                self.graph = partitioned
            return partitioned

        node_sizes = {
            node: len(graph.nodes[node].get('qubits', []))
            for node in graph.nodes
        }
        if any(size != 1 for size in node_sizes.values()):
            raise ValueError(
                "Kernighan-Lin partitioning must be applied to an uncontracted "
                "QIG (one logical qubit per node)."
            )

        qubit_num = graph.number_of_nodes()
        if sum(mems) < qubit_num:
            raise ValueError("Total QPU memory is less than the number of logical qubits.")

        # Allocate a capacity-feasible target size to each non-empty partition.
        # Filling the least utilized QPU first balances utilization while still
        # supporting heterogeneous QPU capacities and spare total capacity.
        capacities = list(mems)
        target_sizes = [0] * len(capacities)
        for _ in range(qubit_num):
            candidates = [
                idx for idx, capacity in enumerate(capacities)
                if target_sizes[idx] < capacity
            ]
            idx = min(
                candidates,
                key=lambda candidate: (
                    target_sizes[candidate] / capacities[candidate],
                    -capacities[candidate],
                    candidate,
                ),
            )
            target_sizes[idx] += 1
        target_sizes = sorted(
            (size for size in target_sizes if size > 0),
            reverse=True,
        )

        rng = np.random.default_rng(seed)

        def recursive_partition(subgraph: nx.Graph, sizes: list[int]) -> list[set]:
            if len(sizes) == 1:
                return [set(subgraph.nodes)]

            # Split the requested partition sizes into two groups whose total
            # vertex counts are as even as possible.
            total = sum(sizes)
            prefix = 0
            split_at = 1
            best_difference = total
            for idx in range(1, len(sizes)):
                prefix += sizes[idx - 1]
                difference = abs(total - 2 * prefix)
                if difference < best_difference:
                    best_difference = difference
                    split_at = idx

            left_sizes = sizes[:split_at]
            right_sizes = sizes[split_at:]
            left_count = sum(left_sizes)

            nodes = list(subgraph.nodes)
            rng.shuffle(nodes)
            initial_partition = (set(nodes[:left_count]), set(nodes[left_count:]))

            if subgraph.number_of_edges() == 0:
                left, right = initial_partition
            else:
                left, right = nx.community.kernighan_lin_bisection(
                    subgraph,
                    partition=initial_partition,
                    max_iter=max_iter,
                    weight='demand',
                    seed=seed,
                )
                # NetworkX labels the two returned sides independently of the
                # order in ``initial_partition``.  Restore the requested side
                # sizes before recursing, which matters for heterogeneous QPUs.
                if len(left) != left_count:
                    left, right = right, left
                if len(left) != left_count:
                    raise RuntimeError("Kernighan-Lin changed the requested partition sizes.")

            return (
                recursive_partition(subgraph.subgraph(left).copy(), left_sizes)
                + recursive_partition(subgraph.subgraph(right).copy(), right_sizes)
            )

        partitions = recursive_partition(graph, target_sizes)

        # Convert the partition into the same superqubit representation used by
        # edge contraction.  Demands between the same two partitions are summed.
        partitioned = nx.Graph()
        membership = {}
        for partition in partitions:
            representative = min(partition)
            qubits = []
            for node in sorted(partition):
                membership[node] = representative
                qubits.extend(graph.nodes[node]['qubits'])
            partitioned.add_node(representative, qubits=qubits)

        for src, dst, data in graph.edges(data=True):
            src_partition = membership[src]
            dst_partition = membership[dst]
            if src_partition == dst_partition:
                continue
            demand = data.get('demand', 0)
            if partitioned.has_edge(src_partition, dst_partition):
                partitioned[src_partition][dst_partition]['demand'] += demand
            else:
                partitioned.add_edge(src_partition, dst_partition, demand=demand)

        if inplace:
            self.graph = partitioned
        return partitioned

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


    def analyze(self):
        """
        Analyze the QIG and print some statistics
        """
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)

        components = list(nx.connected_components(self.graph))
        num_components = len(components)
        component_sizes = [len(c) for c in components]

        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Number of connected components: {num_components}")


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
    # qig = RandomQIG(64, 256, (1, 11))
    # copy_qig = copy.deepcopy(qig)
    qig = QIG.from_qasm('src/circuit/src/qaoa_16.qasm')
    # mems = [16, 16, 12, 12, 8]

    # print(len(qig.graph.edges))
    # print(len(qig.graph.nodes))

    # qig.contract_greedy(8, inplace=True)
    # qig.contract_hdware_constrained(mems, inplace=True)
    # print(len(qig.graph.nodes))

    # count the total qubits on each processor
    # total = 0
    # for node in qig.graph.nodes:
    #     qubit_num = len(qig.graph.nodes[node]['qubits'])
    #     total += qubit_num
    #     print(f"Node {node}: {qubit_num} qubits")
    # print(f"Total qubits: {total}")
    # for edge in qig.graph.edges(data=True):
    #     print('edge:', edge[:2], 'demand:', edge[2]['demand'])

    qig.analyze()
    qig.draw('qig.png')
