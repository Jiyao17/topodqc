
from typing import NewType
import copy

import networkx as nx

# nodes are named by strings in networkx library
Node = NewType('Node', str)
Pair = NewType('Pair', tuple[Node, Node])
Path = NewType('Path', tuple[Node])
Paths = NewType('Paths', list[Path])
PairKSP = NewType('PairKSP', dict[Pair, Paths])


def YenKSP(
        graph: nx.Graph, 
        pairs: list[Pair], 
        k: int=1, 
        weight: str=None, 
        existing_paths: dict=None
        ):
    """
    find k shortest paths between all given pairs
    Parameters
    ----------
    graph: nx.Graph
        The network graph
    pairs: list[Pair], optional
        The pairs of nodes to find paths between
    k: int, optional (default=1)
        The number of shortest paths to find between each pair
    weight: str, optional (default=None)
        -None: least hops
        -'length': Shortest path length
    existing_paths: dict, optional. 
        If provided, those paths are included in the k shortest paths.
    """
    APYenKSP: PairKSP = { pair: [] for pair in pairs }
    
    if existing_paths is not None:
        for pair in existing_paths.keys():
            # slice the existing paths up to k
            APYenKSP[pair] += existing_paths[pair][:k]
    
    for pair in pairs:
        path_iter = nx.shortest_simple_paths(graph, *pair, weight=weight)
        while len(APYenKSP[pair]) < k:
            try:
                path: Path = tuple(next(path_iter))
                APYenKSP[pair].append(path)
            except StopIteration:
                break

    return APYenKSP

def contract_edge(graph: nx.Graph, edge: tuple[Node, Node], inplace: bool=False):
    """
    Similar to nx.contracted_edge, but 
        -attribute values of edges to common neighbors are summed
        -attribute values of nodes are summed
        -remove the self-loop for the merged node
    """
    assert graph.has_edge(*edge), f"Edge {edge} does not exist in the graph"
    
    if not inplace:
        graph = copy.deepcopy(graph)

    # merge the edge attributes
    u, v = edge
    adj_u, adj_v = graph[u], graph[v]
    # find the common neighbors
    common_neighbors = set(adj_u) & set(adj_v)
    # add the edge attributes
    for w in common_neighbors:
        for key in adj_u[w].keys():
            adj_u[w][key] += adj_v[w][key]
    # copy the edges from v to u
    for w in adj_v:
        if w not in common_neighbors and w != u:
            graph.add_edge(u, w, **adj_v[w])
    # merge the nodes' attributes
    for key in graph.nodes[u].keys():
        if key in graph.nodes[v]:
            graph.nodes[u][key] += graph.nodes[v][key]
    # remove the node v
    graph.remove_node(v)

    return graph


if __name__ == "__main__":
    nodes = [1, 2, 3, 4]
    node_attrs = { node: [node] for node in nodes }
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)]
    weights = { edge: 1 for edge in edges }
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_edge_attributes(graph, weights, 'weight')
    nx.set_node_attributes(graph, node_attrs, 'qubits')

    contract_edge(graph, (1, 2), inplace=True)
    layout = nx.spring_layout(graph)
    nx.draw(graph, pos=layout, with_labels=True)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels)
    import matplotlib.pyplot as plt
    plt.savefig('result/contracted.png')
    plt.close()

    print(graph.nodes(data=True))

    contract_edge(graph, (3, 4), inplace=True)
    layout = nx.spring_layout(graph)
    nx.draw(graph, pos=layout, with_labels=True)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels)
    plt.savefig('result/contracted2.png')

    print(graph.nodes(data=True))