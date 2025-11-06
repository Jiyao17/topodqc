
from collections import Counter, defaultdict
import sys
import time 

import numpy as np
from src.circuit.qig import QIG, RandomQIG

from .taco import TACO
from .taco_l import TACOL

from src.utils.graph import draw_topology

class TACOPA(TACOL):
    def __init__(self, qig, mems, comms, W, edge_weights=None, timeout=300):
        super().__init__(qig, mems, comms, W, edge_weights, timeout)

    def solve(self):
        # get node: mem pairs
        nodes = [(node, len(self.qig.graph.nodes[node]['qubits'])) for node in self.qig.graph.nodes]
        qpus = list(self.qpus.items())

        squbits = [(self.squbits_rev[node], len(self.qig.graph.nodes[node]['qubits'])) for node in self.qig.graph.nodes]
        print(squbits)
        print(qpus)

        assignments = {}
        # directly assign nodes to qpus
        while len(nodes) > 0:
            # sort nodes and qpus
            nodes.sort(key=lambda x: x[1], reverse=True)
            qpus.sort(key=lambda x: x[1], reverse=True)
            
            # assign first node to first qpu if possible
            squbit_idx = self.squbits_rev[nodes[0][0]]
            qpu_idx = qpus[0][0]

            qup0_mem = qpus[0][1] - nodes[0][1]
            if qup0_mem < 0:
                raise ValueError("Node cannot be assigned to any processor due to insufficient memory.")

            # assign by constrain
            self.model.addConstr(self.x[squbit_idx, qpu_idx] == 1)
            assignments[squbit_idx] = qpu_idx

            # update nodes and qpus
            nodes = nodes[1:]
            if qup0_mem == 0:
                # remove the qpu if no memory left
                qpus = qpus[1:]
            else:
                qpus[0] = (qpus[0][0], qup0_mem)

        # call parent solve method
        print(len(qpus), len(nodes))
        print(assignments)
        return super().solve()
    



def form_racks(
        mems: list[int], rack_size: int, 
        in_rack_cost: float, cross_rack_cost: float
        ) -> dict[tuple[int, int], float]:
    
    # map from mem to list of proc ids
    qpus = defaultdict(list)
    for i, mem in enumerate(mems):
        qpus[mem].append(i)

    # put only processors with same memory size in the same rack
    # form each rack greedily
    # remaining processors are put in the last rack
    racks = []
    # from proc id to rack id
    get_rack = {}
    for mem, procs in qpus.items():
        rack_num = len(procs) // rack_size
        for r in range(rack_num):
            rack_procs = procs[r*rack_size:(r+1)*rack_size]
            racks.append(rack_procs)
            for p in rack_procs:
                get_rack[p] = len(racks) - 1
        # remaining procs
        rem_procs = procs[rack_num*rack_size:]
        if len(rem_procs) > 0:
            racks.append(rem_procs)
            for p in rem_procs:
                get_rack[p] = len(racks) - 1

    # form edge weights
    edge_weights = {}
    for i in range(len(mems)):
        for j in range(i+1, len(mems)):
            if get_rack[i] == get_rack[j]:
                edge_weights[(i, j)] = in_rack_cost
            else:
                edge_weights[(i, j)] = cross_rack_cost

    return edge_weights

def form_ANL() -> dict[tuple[int, int], float]:
    """
    Form the ANL topology
    """
    sites = {
        0: 'NU-Evanston',
        1: 'StarLight',
        2: 'UChicago-PME',
        3: 'UChicago-HC',
        4: 'Fermilab-1',
        5: 'Fermilab-2',
        6: 'Argonne-1',
        7: 'Argonne-2',
        8: 'Argonne-3',
    }

    distances = {
        (0, 1): 20,
        (1, 2): 16,
        (1, 4): 66,
        (1, 6): 54,
        (2, 3): 2,
        (2, 6): 42,
        (4, 5): 2,
        (4, 6): 53,
        (6, 7): 0.1,
        (6, 8): 41.8,
    }
    
    
    edge_weights = {}
    for i in range(len(sites)):
        for j in range(i+1, len(sites)):
            if (i, j) in distances:
                edge_weights[(i, j)] = pow(2, distances[(i, j)])
            else:
                # max value for non-existing links
                # edge_weights[(i, j)] = sys.float_info.max
                edge_weights[(i, j)] = 1e100

    return edge_weights



if __name__ == "__main__":
    np.random.seed(0)
    # proc_num = 8
    # mem = 8
    # comm = 4

    qubit_num = 128
    # demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = int(qubit_num * (qubit_num-1) / 6) # max
    demand_pair = qubit_num * 4 # moderate

    qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # print(sorted(qig.demands))
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig = QIG.from_qasm('src/circuit/src/grover_256.qasm')
    # qig = QIG.from_qasm('src/circuit/src/mcmt_128c_128t.qasm')

    # qig.contract(4, inplace=True)

    # 1024, 570 second to contract
    # mems = [256, 256, 128, 128, 64, 64, 64, 64]
    # comms = [64, 64, 32, 32, 16, 16, 16, 16]
    # 512, need 40 seconds to contract
    # mems = [128, 128, 64, 64, 32, 32, 32, 32]
    # comms = [32, 32, 16, 16, 8, 8, 8, 8]
    # 256, need 3.33 seconds to contract
    # mems = [64, 64, 32, 32, 16, 16, 16, 16]
    # comms = [16, 16, 8, 8, 4, 4, 4, 4]
    # 256 homogeneous
    mems = [32, ] * 9
    comms = [8, ] * 9
    W = 100



    # mems = [32, 32, 16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 8, 8, 4, 4, 4, 4]
    # mems = [16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 4, 4, 4, 4]
    # mems = [2, ] * 8
    # comms = [8, ] * 8
    # W = 160
    # W = int(proc_num * (proc_num-1) / 2)
    # W = 1e6
    # W = (proc_num - 1)
    # W = proc_num  + 1

    # edge_weights = form_racks(
    #     mems, rack_size=4, 
    #     in_rack_cost=1.0, cross_rack_cost=1000.0
    #     )
    edge_weights = form_ANL()
    print(edge_weights)

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    # qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    model = TACOPA(qig, mems, comms, W, edge_weights)
    # model = TACOPA(qig, mems, comms, W)
    model.build()
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    print("Solving the model...")
    start_time = time.time()
    print(model.solve())
    print(f"Model solved in {time.time() - start_time:.2f} seconds.")

    edges = model.get_topology()
    # print(edges)
    draw_topology(edges, filename='result/taco_pa_topology.png')
    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)


    # for a in model.qubits:
    #     for i in model.procs:
    #         print((a, i), model.x[a, i].x)
    
    # path_lengths = model.get_results()
    # print(path_lengths)