    graphes = copy_qig.contract_all_branches(2, 8, 100)
    graphes = list(graphes.values())
    graphes = sorted(graphes, key=lambda g: len(g.nodes))
    # graphes = graphes[:100]  # limit to 100 graphes
    print(f"Total contracted graphes: {len(graphes)}")
    # pick the best graphes with the least number of nodes
    min_nodes = float('inf')
    best_graph = None
    for i, graph in enumerate(graphes):
        total_nodes = len(graph.nodes)
        # print(f"Graph {i}: Total demand = {total_demand}")
        if total_nodes < min_nodes:
            min_nodes = total_nodes
            best_graph = graph
    print(f"Best graph total nodes: {min_nodes}")

    edge_labels = nx.get_edge_attributes(best_graph, 'demand')
    node_labels = nx.get_node_attributes(best_graph, 'qubits')
    node_labels = {node: len(qubits) for node, qubits in node_labels.items()}
    draw(best_graph, edge_labels=edge_labels, node_labels=node_labels, filename='best_qig.png')

    # draw the top 10 graphes
    for i, graph in enumerate(graphes[:10]):
        edge_labels = nx.get_edge_attributes(graph, 'demand')
        node_labels = nx.get_node_attributes(graph, 'qubits')
        node_labels = {node: len(qubits) for node, qubits in node_labels.items()}
        draw(graph, edge_labels=edge_labels, node_labels=node_labels, filename=f'graph_{i}.png')

        # print total demands
        total_demand = sum(edge[2]['demand'] for edge in graph.edges(data=True))
        print(f"Graph {i}: Total demand = {total_demand}")