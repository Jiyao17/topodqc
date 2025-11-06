
"""
Application of TACOPA to solve
1. quantum data center networks (Top on Rack switched networks) (SwitchQN)
2. Argonne National Lab network (ANL) in Chicago area (SeQUeNCe)

"""



from collections import defaultdict
import time

import numpy as np

from src.circuit.qig import QIG, RandomQIG
from src.solver.taco_pa import TACOPA
from src.solver.taco_l import TACOL
from src.utils.graph import draw_topology


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
                edge_weights[(i, j)] = 1e15

    return edge_weights




def test_SwitchQN():
    np.random.seed(0)
    # proc_num = 8
    # mem = 8
    # comm = 4

    # qubit_num = 1024
    # demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = int(qubit_num * (qubit_num-1) / 6) # max
    # demand_pair = qubit_num * 4 # moderate

    # qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # print(sorted(qig.demands))
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig = QIG.from_qasm('src/circuit/src/grover_256.qasm')
    # qig = QIG.from_qasm('src/circuit/src/mcmt_128c_128t.qasm')
    qig = QIG.from_qasm('src/circuit/src/qft_512.qasm')
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
    mems = [64, ] * 16
    comms = [2, ] * 16
    W = 40



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

    edge_weights = form_racks(
        mems, rack_size=4, 
        in_rack_cost=1, cross_rack_cost=100*8
        )
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


def test_ANL():
    # np.random.seed(0)

    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig = QIG.from_qasm('src/circuit/src/grover_256.qasm')
    # qig = QIG.from_qasm('src/circuit/src/mcmt_128c_128t.qasm')
    # qig = QIG.from_qasm('src/circuit/src/qft_1024.qasm')
    qig = QIG.from_qasm('src/circuit/src/qft_1536.qasm')
    # qig = QIG.from_qasm('src/circuit/src/qft_2048.qasm')

    # qig.contract(4, inplace=True)


    mems = [256, ] * 9
    comms = [16, ] * 9
    W = 10

    edge_weights = form_ANL()
    print(edge_weights)

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    # model = TACOPA(qig, mems, comms, W, edge_weights)
    model = TACOL(qig, mems, comms, W, edge_weights)
    # model = TACOPA(qig, mems, comms, W)
    model.build()
    # model.model.addConstr(model.w[6, 7] == 1, name='fix_link_6_7')
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


if __name__ == "__main__":
    # test_ANL()
    test_SwitchQN()