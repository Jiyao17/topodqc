
import numpy as np
import pickle
import time

from src.circuit import QIG, QASM_FILES, RandomQIG
from src.solver import TACO, TACOORIG, TACONL, TACOL


def test_solver(SolverClass, qig: QIG, mems: list, comms: list, W: int, timeout: int=600):
    assert SolverClass in [TACOORIG, TACONL, TACOL]
    solver: TACO = SolverClass(qig, mems, comms, W, timeout)
    solver.build()
    solver.solve()
    
    objs = solver.get_objs()
    edges = solver.get_topology()

    return objs, edges
    
def run_tests_real(folder = 'result/'):
    timeout = 300
    np.random.seed(0)
    qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]
    # qbits = [16]
    # pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    # print(pairs)
    # print(cnot_nums)

    # rand_qigs = [ 
    #     RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    # ]
    qigs = qigs[:1]
    qigs_names = [ "adr4", "clip", "co14" ]
    # qigs_names = [ "rand16" ]

    clusters = [
        # (4, 4, 4, 4),
        # (8, 2, 2, 8),
        (8, 2, 4, 8),
        # (6, 4, 4, 6),
        # (6, 4, 6, 8),
        # (8, 4, 4, 10),
        # (8, 4, 6, 12),
        # (8, 8, 8, 16),
    ]

    # solvers = [ TACOORIG, TACONL, TACOL ]

    objs = {}
    for qig, name in zip(qigs, qigs_names):
        cluster = clusters[0]
        n_qpu, n_mem, n_comm, W = cluster
        mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu
        capacity = n_qpu * n_mem

        print(f'Simulations for cluster: {cluster} running...')
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
        
        with open(f'{folder}objs-real-adr4.pkl', 'wb') as f:
            pickle.dump(objs, f)

    
def run_tests_cluster(folder = 'result/'):
    timeout = 120
    np.random.seed(0)
    qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]
    # qbits = [16]
    # pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    # print(pairs)
    # print(cnot_nums)

    # rand_qigs = [ 
    #     RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    # ]
    qigs = qigs[:1]
    qigs_names = [ "adr4", "clip", "co14" ]
    # qigs_names = [ "rand16" ]

    clusters = [
        # (4, 4, 4, 4),
        # (8, 2, 2, 8),
        (8, 2, 4, 8),
        # (6, 4, 4, 6),
        # (6, 4, 6, 8),
        (8, 3, 4, 8),
        (8, 4, 4, 8),
        # (8, 4, 6, 12),
        # (8, 8, 8, 16),
    ]

    # solvers = [ TACOORIG, TACONL, TACOL ]

    objs = {}
    for qig, name in zip(qigs, qigs_names):
        cluster = clusters[0]
        n_qpu, n_mem, n_comm, W = cluster
        mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu
        capacity = n_qpu * n_mem

        print(f'Simulations for cluster: {cluster} running...')
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
        
        with open(f'{folder}objs-real-adr4-cluster.pkl', 'wb') as f:
            pickle.dump(objs, f)


def run_tests_rand(folder = 'result/'):
    timeout = 300
    np.random.seed(0)
    qbits = [24, 32]
    pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    # print(pairs)
    # print(cnot_nums)

    rand_qigs = [ 
        RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    ]
    qigs = rand_qigs
    # qigs_names = [ "adr4", "clip", "co14" ]
    # qigs_names = [ "rand16" ]
    qigs_names = [ "rand24", "rand32" ]

    clusters = [
        # (4, 4, 4, 4),
        # (8, 2, 2, 8),
        # (8, 2, 4, 10),
        # (6, 4, 4, 6),
        # (12, 2, 4, 14),
        (8, 3, 4, 10),
        (8, 4, 4, 12),
        # (16, 2, 4, 20),
    ]

    # solvers = [ TACOORIG, TACONL, TACOL ]

    objs = {}
    for qig, name, cluster in zip(qigs, qigs_names, clusters):
        n_qpu, n_mem, n_comm, W = cluster
        mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu
        capacity = n_qpu * n_mem

        print(f'Simulations for cluster: {cluster} running...')
        n_qubits = len(qig.graph.nodes)
        if n_qubits > capacity:
            objs[(cluster, qig, TACOORIG)] = None
            objs[(cluster, qig, TACONL)] = None
            objs[(cluster, qig, TACOL)] = None

        orig_objs, orig_topo = test_solver(TACOORIG, qig, mems, comms, W, timeout)
        nl_objs, nl_topo = test_solver(TACONL, qig, mems, comms, W, timeout)
        l_objs, l_topo = test_solver(TACOL, qig, mems, comms, W, timeout)

        objs[(cluster, name, TACOORIG)] = orig_objs
        objs[(cluster, name, TACONL)] = nl_objs
        objs[(cluster, name, TACOL)] = l_objs
        
        with open(f'{folder}objs-ml.pkl', 'wb') as f:
            pickle.dump(objs, f)

      
def test(folder = 'result/'):
    timeout = 180
    np.random.seed(0)
    qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]
    qigs = qigs[:1]
    qigs_names = [ "adr4", "clip", "rand16", "rand24", "rand32" ]

    clusters = [
        # (4, 4, 4, 4),
        (8, 2, 2, 12),
    ]
    # capacities = [ cluster[0]*cluster[1] for cluster in clusters ]
    # print(capacities)

    solvers = [ TACOORIG, TACONL, TACOL ]
    qpus = []

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

        with open(f'{folder}objs.pkl', 'wb') as f:
            pickle.dump(objs, f)
        with open(f'{folder}topos.pkl', 'wb') as f:
            pickle.dump(topos, f)
            
        print(f'Simulations for cluster: {cluster} done!')

if __name__ == "__main__":

    run_tests_real()
    run_tests_rand()
    run_tests_cluster()
    # test()
