
import time 

import numpy as np
from src.circuit.qig import QIG, RandomQIG

from .taco_l import TACOL

class TACOPA(TACOL):
    def __init__(self, qig, mems, comms, W, timeout=600):
        super().__init__(qig, mems, comms, W, timeout)

    def solve(self):
        # sort nodes by number of qubits
        nodes = sorted(self.qig.graph.nodes, key=lambda n: len(self.qig.graph.nodes[n]['qubits']), reverse=True)
        qpus = list(self.qpus.items())
        # sort qupus by memory size
        qpus.sort(key=lambda x: x[1], reverse=True)

        # directly assign nodes to qpus sequentially
        while len(nodes) > 0 and len(qpus) > 0:
            new_qpus = []
            for node, qpu in zip(nodes, qpus):
                node_idx = self.squbits_rev[node]
                qpu_idx = qpu[0]

                # assign by constrain
                self.model.addConstr(self.x[node_idx, qpu_idx] == 1)

                # remove assigned capacity from qpu
                if qpu[1] > self.squbits_sizes[node_idx]:
                    new_qpus.append((qpu[0], qpu[1] - self.squbits_sizes[node_idx]))

            # update nodes and qpus
            # nodes are those not assigned
            if len(nodes) > len(qpus):
                nodes = nodes[len(qpus):]
            qpus = new_qpus
            qpus.sort(key=lambda x: x[1], reverse=True)

        # call parent solve method
        return super().solve()
    

if __name__ == "__main__":
    np.random.seed(0)
    proc_num = 8
    mem = 8
    comm = 4

    qubit_num = 256
    demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = int(qubit_num * (qubit_num-1) / 6) # max
    # demand_pair = qubit_num * 2 # moderate

    qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # print(sorted(qig.demands))
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
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
    mems = [32, ] * 8
    comms = [8, ] * 8

    # mems = [32, 32, 16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 8, 8, 4, 4, 4, 4]
    # mems = [16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 4, 4, 4, 4]
    # mems = [8, 8, 4, 4, 4, 4]
    # comms = [4, 4, 4, 4, 4, 4]
    W = int(proc_num * (proc_num-1) / 2)
    # W = 1e6
    # W = (proc_num - 1)
    # W = proc_num  + 1

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    model = TACOPA(qig, mems, comms, W)
    model.build()
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    print("Solving the model...")
    start_time = time.time()
    print(model.solve())
    print(f"Model solved in {time.time() - start_time:.2f} seconds.")


    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)


    # for a in model.qubits:
    #     for i in model.procs:
    #         print((a, i), model.x[a, i].x)
    
    # path_lengths = model.get_results()
    # print(path_lengths)