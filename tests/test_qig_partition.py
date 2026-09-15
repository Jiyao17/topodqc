import unittest
import copy

import networkx as nx

from src.circuit.qig import QIG
from src.utils.graph import contract_edge


class KernighanLinPartitionTests(unittest.TestCase):
    @staticmethod
    def make_two_cluster_qig() -> QIG:
        qig = QIG()
        qig.qubits = list(range(6))
        qig.demands = {
            (0, 1): 10,
            (0, 2): 10,
            (1, 2): 10,
            (3, 4): 10,
            (3, 5): 10,
            (4, 5): 10,
            (2, 3): 1,
        }
        qig.graph = qig.make_graph()
        return qig

    def test_partition_preserves_qubits_and_capacity(self):
        qig = self.make_two_cluster_qig()

        graph = qig.partition_kernighan_lin([3, 3], seed=7)

        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(
            sorted(len(graph.nodes[node]['qubits']) for node in graph.nodes),
            [3, 3],
        )
        assigned_qubits = [
            qubit
            for node in graph.nodes
            for qubit in graph.nodes[node]['qubits']
        ]
        self.assertEqual(sorted(assigned_qubits), list(range(6)))
        self.assertEqual(sum(nx.get_edge_attributes(graph, 'demand').values()), 1)

    def test_inplace_partition_supports_heterogeneous_spare_capacity(self):
        qig = self.make_two_cluster_qig()

        returned = qig.partition_kernighan_lin([5, 2], inplace=True)

        self.assertIs(returned, qig.graph)
        sizes = sorted(
            (len(qig.graph.nodes[node]['qubits']) for node in qig.graph.nodes),
            reverse=True,
        )
        self.assertEqual(sizes, [4, 2])

    def test_partition_rejects_an_already_contracted_qig(self):
        qig = self.make_two_cluster_qig()
        qig.graph.nodes[0]['qubits'].append(6)

        with self.assertRaisesRegex(ValueError, 'uncontracted'):
            qig.partition_kernighan_lin([4, 3])

    def test_recursive_partition_keeps_heterogeneous_target_sizes(self):
        qig = QIG()
        qig.qubits = list(range(12))
        qig.demands = {(node, node + 1): 1 for node in range(11)}
        qig.graph = qig.make_graph()

        graph = qig.partition_kernighan_lin([5, 4, 3])

        self.assertEqual(
            sorted(
                (len(graph.nodes[node]['qubits']) for node in graph.nodes),
                reverse=True,
            ),
            [5, 4, 3],
        )


class EdgeContractionOptimizationTests(unittest.TestCase):
    @staticmethod
    def reference_contraction(graph: nx.Graph, mems: list[int]) -> nx.Graph:
        """Original EC implementation retained as a parity oracle."""
        graph = copy.deepcopy(graph)

        def comply(squbits):
            capacities = copy.deepcopy(mems)
            squbits = copy.deepcopy(squbits)
            while squbits:
                capacities.sort(reverse=True)
                squbits.sort(reverse=True)
                if squbits[0] > capacities[0]:
                    return False
                capacities[0] -= squbits.pop(0)
                if capacities[0] == 0:
                    capacities.pop(0)
            return True

        while True:
            edges = sorted(
                graph.edges(data=True),
                key=lambda edge: edge[2]['demand'],
                reverse=True,
            )
            candidate = None
            for edge in edges:
                p1, p2 = edge[:2]
                sizes = {
                    node: len(graph.nodes[node]['qubits'])
                    for node in graph.nodes
                }
                sizes[p1] += sizes[p2]
                del sizes[p2]
                if comply(list(sizes.values())):
                    candidate = edge
                    break
            if candidate is None:
                return graph
            contract_edge(graph, candidate[:2], inplace=True)

    @staticmethod
    def partition_signature(graph: nx.Graph):
        return {
            frozenset(graph.nodes[node]['qubits'])
            for node in graph.nodes
        }

    def test_optimized_ec_matches_original_greedy_decisions(self):
        for seed, mems in ((3, [4, 4, 4]), (8, [6, 4, 2])):
            qig = QIG()
            qig.qubits = list(range(12))
            graph = nx.gnp_random_graph(12, 0.35, seed=seed)
            # Unique weights make the greedy edge ordering unambiguous.
            for demand, edge in enumerate(graph.edges, start=1):
                graph.edges[edge]['demand'] = demand
            nx.set_node_attributes(
                graph,
                {node: [node] for node in graph.nodes},
                'qubits',
            )
            qig.graph = graph

            expected = self.reference_contraction(qig.graph, mems)
            actual = qig.contract_hdware_constrained(mems)

            self.assertEqual(
                self.partition_signature(actual),
                self.partition_signature(expected),
            )

    def test_optimized_ec_returns_inplace_graph(self):
        qig = KernighanLinPartitionTests.make_two_cluster_qig()

        returned = qig.contract_hdware_constrained([3, 3], inplace=True)

        self.assertIs(returned, qig.graph)


if __name__ == '__main__':
    unittest.main()
