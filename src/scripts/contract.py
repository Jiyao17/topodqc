

import numpy as np

from src.circuit.qig import QIG, QASM_FILES, RandomQIG
from src.solver.taco_l import TACOL
from src.scripts.runtime import test_solver


def comp_contract():
    timeout = 300
    np.random.seed(0)
    # qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]
    qbits = [512]
    pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    # print(pairs)
    # print(cnot_nums)

    rand_qigs = [ 
        RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    ]
    # qigs = qigs[:1]
    qigs = rand_qigs[:1]

    # qigs_names = [ "adr4", "clip", "co14" ]
    qigs_names = [ "rand16" ]

    clusters = [
        # (4, 4, 4, 4),
        # (8, 2, 2, 8),
        # (8, 2, 4, 8),
        # (6, 4, 4, 6),
        # (6, 4, 6, 8),
        # (8, 4, 4, 10),
        # (8, 4, 6, 12),
        # (8, 8, 8, 16),
        # (8, 16, 8, 64),
        (16, 32, 8, 64),
        # (32, 32, 8, 64),
    ]

    for qig, name in zip(qigs, qigs_names):
        cluster = clusters[0]
        n_qpu, n_mem, n_comm, W = cluster
        mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu

        print(f'Simulations for cluster: {cluster} running...')

        qig.contract_greedy(mem_limit=n_mem, inplace=True)

        print("QIG size after contraction: ")
        print(len(qig.graph.nodes), len(qig.graph.edges))

        objs, topo = test_solver(TACOL, qig, mems, comms, W, timeout)
        # objs_con, topo_con = test_solver(TACOL, qig, mems, comms, W, timeout)



if __name__ == "__main__":
    comp_contract()