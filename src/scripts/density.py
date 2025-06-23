

import numpy as np
import pickle
import time

from src.circuit import QIG, QASM_FILES, RandomQIG
from src.solver import TACO, TACOORIG, TACONL, TACOL
from src.scripts.runtime import test_solver

def run_tests_small(folder = 'result/density/'):
    timeout = 120
    np.random.seed(0)
    qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]

    qbits = [24, 32]
    pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    rand_qigs = [ 
        RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    ]
    qigs = qigs[:1] + rand_qigs

    qigs_names = [ "adr4", "rand24", "rand32" ]
    # qigs_names = [ "adr4", "clip", "co14", "rand16",]
    # circ_name = "rand24"
    # qigs_names = [ circ_name,]
    clusters = [
        [
        (8, 2, 4, 8),
        (8, 2, 4, 10),
        (8, 2, 4, 12),
        ],
        [
        (8, 3, 4, 8),
        (8, 3, 4, 10),
        (8, 3, 4, 12),
        ],        
        [
        (8, 4, 4, 8),
        (8, 4, 4, 10),
        (8, 4, 4, 12),
        ],    
    ]

    # solvers = [ TACOORIG, TACONL, TACOL ]

    objs = {}
    topos = {}
    for i, qigs_name in enumerate(qigs_names):
        cluster = clusters[i]
        for c in cluster:
            n_qpu, n_mem, n_comm, W = c
            mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu
            capacity = n_qpu * n_mem
            
            for qig, name in zip(qigs, qigs_names):
                n_qubits = len(qig.graph.nodes)
                if n_qubits > capacity:
                    # objs[(cluster, name, TACOORIG)] = None
                    # objs[(cluster, name, TACONL)] = None
                    objs[(c, name, TACOL)] = None

                # orig_objs, orig_topo = test_solver(TACOORIG, qig, mems, comms, W, timeout)
                # nl_objs, nl_topo = test_solver(TACONL, qig, mems, comms, W, timeout)
                l_objs, l_topo = test_solver(TACOL, qig, mems, comms, W, timeout)

                # objs[(cluster, name, TACOORIG)] = orig_objs
                # objs[(cluster, name, TACONL)] = nl_objs
                objs[(c, name, TACOL)] = l_objs
                # topos[(cluster, name, TACOORIG)] = orig_topo
                # topos[(cluster, name, TACONL)] = nl_topo
                topos[(c, name, TACOL)] = l_topo
            

            with open(f'{folder}objs-{qigs_name}.pkl', 'wb') as f:
                pickle.dump(objs, f)
            


def test(folder = 'result/'):
    timeout = 180
    np.random.seed(0)
    qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]
    qbits = [16]
    pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    # print(pairs)
    # print(cnot_nums)

    rand_qigs = [ 
        RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    ]
    qigs = qigs + rand_qigs
    # qigs_names = [ "adr4", "clip", "co14", "rand16", "rand32" ]
    qigs_names = [ "adr4"]

    clusters = [
        # (4, 4, 4, 4),
        (8, 2, 2, 12),
        (8, 2, 2, 14),
        (8, 2, 2, 16),
        # (8, 4, 4, 16),
        # (8, 8, 8, 16),
    ]

    # solvers = [ TACOORIG, TACONL, TACOL ]

    objs = {}
    topos = {}
    for cluster in clusters:
        n_qpu, n_mem, n_comm, W = cluster
        mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu
        capacity = n_qpu * n_mem
        print(f'Simulations for cluster: {cluster} running...')
        for qig, name in zip(qigs, qigs_names):
            n_qubits = len(qig.graph.nodes)
            if n_qubits > capacity:
                objs[(cluster, name, TACOORIG)] = None
                objs[(cluster, name, TACONL)] = None
                objs[(cluster, name, TACOL)] = None

            orig_objs, orig_topo = test_solver(TACOORIG, qig, mems, comms, W, timeout)
            nl_objs, nl_topo = test_solver(TACONL, qig, mems, comms, W, timeout)
            l_objs, l_topo = test_solver(TACOL, qig, mems, comms, W, timeout)

            objs[(cluster, name, TACOORIG)] = orig_objs
            objs[(cluster, name, TACONL)] = nl_objs
            objs[(cluster, name, TACOL)] = l_objs
            topos[(cluster, name, TACOORIG)] = orig_topo
            topos[(cluster, name, TACONL)] = nl_topo
            topos[(cluster, name, TACOL)] = l_topo
            

        # with open(f'{folder}objs.pkl', 'wb') as f:
        #     pickle.dump(objs, f)
        with open(f'{folder}topos-small.pkl', 'wb') as f:
            pickle.dump(topos, f)

        print(f'Simulations for cluster: {cluster} done!')


if __name__ == "__main__":

    run_tests_small()
    # run_tests_medium()
    # run_tests_large()
    # test()
