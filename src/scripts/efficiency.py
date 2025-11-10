
import os
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt

from src.circuit import QIG, QASM_FILES, RandomQIG
from src.solver import TACO, TACOORIG, TACONL, TACOL, TACOSA, TACOPA


def test_solver(SolverClass, qig: QIG, mems: list, comms: list, W: int, edge_weights: dict[tuple[int, int], float]=None, timeout: int=600):
    assert SolverClass in [TACOORIG, TACONL, TACOL, TACOSA, TACOPA]
    if SolverClass == TACOL or SolverClass == TACOPA:
        solver: TACO = SolverClass(qig, mems, comms, W, edge_weights, timeout)
    else:
        solver: TACO = SolverClass(qig, mems, comms, W, timeout)

    solver.build()
    solver.solve()
    
    objs = solver.get_objs()
    edges = solver.get_topology()

    return objs, edges


def run_tests(folder = 'result/efficiency/'):
    timeout = 300
    np.random.seed(42)

    tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'QFT' ]
    # tasks = [ 'Grover' ]
    # tasks = [ 'MCMT' ]

    # sizes = [ 4, 8, ]
    # sizes = [ 32, 48, 64, 96 ]
    # sizes = [ 24, 32 ]
    # sizes = [ 64, 96 ]
    # sizes = [ 16 ]
    # sizes = [ 24 ]
    # sizes = [ 32 ]
    sizes = [ 96 ]

    clusters = {
        4: ([2,]*2, [2,]*2, 1),
        8: ([4,]*2, [3,]*2, 1),
        16: ([8,]*2, [3,]*4, 5),
        24: ([4,]*6, [3,]*6, 8),
        32: ([8,]*4, [3,]*4, 5),
        48: ([8,]*4 + [4,]*4, [3,]*8, 10),
        64: ([8,]*8, [3,]*8, 10),
        96: ([8,]*8 + [4,]*8, [3,]*16, 20),
    }

    # clusters = {
    #     16: (4, 4, 2, 5),
    #     24: (4, 6, 3, 5),
    #     32: (4, 8, 4, 5),
    # }


    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    SolverClasses = [ TACOL, TACOORIG, TACONL,  ]
    # SolverClasses = [ TACOL ]
    # SolverClasses = [ TACOSA ]

    for task in tasks:
        for size in sizes:
            qasm_file = QASM_FILES[task, size]
            qig = QIG.from_qasm(qasm_file)
            cluster = clusters[size]
            mems, comms, W = cluster

            for SolverClass in SolverClasses:
                print(f'Simulations for {task}-{size} with {SolverClass.__name__} running...')
                n_qubits = len(qig.graph.nodes)

                objs, topos = test_solver(SolverClass, qig, mems, comms, W, None, timeout)

                with open(f'{folder}objs-{task}-{size}-{SolverClass.__name__}.pkl', 'wb') as f:
                    pickle.dump(objs, f)


                print(f'Simulations for {task}-{size} with {SolverClass.__name__} done!')

                # print("Objs:", objs)


def run_tests_contracted(folder = 'result/efficiency/'):
    timeout = 300
    np.random.seed(42)

    # tasks = [ 'MCMT', 'QFT', 'Grover', ]
    tasks = [ 'MCMT' ]
    # tasks = [ 'QFT', 'Grover' ]
    # tasks = [ 'Grover' ]

    # sizes = [ 32, 48, 64, 96 ]
    # sizes = [ 24, 32 ]
    # sizes = [ 64, 96 ]
    # sizes = [ 16 ]
    # sizes = [ 24 ]
    # sizes = [ 32 ]
    # sizes = [ 256, 384, 512, 768, 1024 ]
    # sizes = [ 256,]
    # sizes = [ 384, ]
    sizes = [ 512, ]
    # sizes = [ 768, ]
    # sizes = [ 1024 ]



    clusters = {
        16: ([4,]*4, [3,]*4, 5),
        24: ([4,]*6, [3,]*6, 8),
        32: ([8,]*4, [3,]*4, 5),
        48: ([8,]*4 + [4,]*4, [3,]*8, 10),
        64: ([8,]*8, [3,]*8, 10),
        96: ([8,]*8 + [4,]*8, [3,]*16, 20),
        256: ([16,]*16, [4,]*16, 40),
        384: ([16,]*16 + [8,]*16, [4,]*16 + [3,]*16, 80),
        512: ([32,]*16, [8,]*16, 40),
        768: ([32,]*16 + [16,]*16, [8,]*16 + [4,]*16, 80),
        1024: ([32,]*32, [8,]*32, 80),
    }




    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    # SolverClasses = [ TACOL, TACOORIG, TACONL,  ]
    # SolverClasses = [ TACOPA, TACOL ]
    # SolverClasses = [ TACOPA, ]
    SolverClasses = [ TACOL, ]
    # SolverClasses = [ TACOSA ]

    for task in tasks:
        for size in sizes:
            qasm_file = QASM_FILES[task, size]
            qig = QIG.from_qasm(qasm_file)
            cluster = clusters[size]
            mems, comms, W = cluster
            qig.contract_hdware_constrained(mems=mems, inplace=True)

            for SolverClass in SolverClasses:
                print(f'Simulations for {task}-{size} with {SolverClass.__name__} running...')
                n_qubits = len(qig.graph.nodes)
                print(f'QIG size after contraction: {n_qubits} squbits, {len(qig.graph.edges)} edges')
                print(f"Sizes of squbits: {sorted([len(qig.graph.nodes[node]['qubits']) for node in qig.graph.nodes], reverse=True)}")

                objs, topos = test_solver(SolverClass, qig, mems, comms, W, None, timeout)

                with open(f'{folder}objs-{task}-{size}-{SolverClass.__name__}.pkl', 'wb') as f:
                    pickle.dump(objs, f)


                print(f'Simulations for {task}-{size} with {SolverClass.__name__} done!')

                # print("Objs:", objs)


def run_contraction(folder = 'result/efficiency/'):
    timeout = 300
    np.random.seed(42)

    tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'QFT' ]
    # tasks = [ 'Grover' ]
    # tasks = [ 'MCMT' ]

    # sizes = [ 32, 48, 64, 96 ]
    # sizes = [ 24, 32 ]
    # sizes = [ 64, 96 ]
    # sizes = [ 16 ]
    # sizes = [ 24 ]
    # sizes = [ 32 ]
    sizes = [ 192, ]
    # sizes = [ 256, 384, 512, 768, 1024 ]
    # sizes = [ 256, 384, 512, ]
    # sizes = [ 768, 1024 ]




    clusters = {
        16: ([4,]*4, [3,]*4, 5),
        24: ([4,]*6, [3,]*6, 8),
        32: ([8,]*4, [3,]*4, 5),
        48: ([8,]*4 + [4,]*4, [3,]*8, 10),
        64: ([8,]*8, [3,]*8, 10),
        96: ([8,]*8 + [4,]*8, [3,]*16, 20),
        192: ([16,]*8 + [8,]*8, [4,]*8 + [3,]*8, 20),
        256: ([16,]*16, [4,]*16, 20),
        384: ([16,]*16 + [8,]*16, [4,]*16 + [3,]*16, 40),
        512: ([32,]*16, [8,]*16, 20),
        768: ([32,]*16 + [16,]*16, [8,]*16 + [4,]*16, 40),
        1024: ([32,]*32, [8,]*32, 40),
    }




    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    # SolverClasses = [ TACOL, TACOORIG, TACONL,  ]
    # SolverClasses = [ TACOPA, TACOL ]
    # SolverClasses = [ TACOSA ]

    result_file = f'{folder}contraction_times.pkl'
    contract_time = {}
    if os.path.exists(result_file):
        with open(result_file, 'rb') as f:
            contract_time = pickle.load(f)

    for task in tasks:
        for size in sizes:
            qasm_file = QASM_FILES[task, size]
            qig = QIG.from_qasm(qasm_file)
            cluster = clusters[size]
            mems, comms, W = cluster
            time_start = time.time()
            qig.contract_hdware_constrained(mems=mems, inplace=True)
            time_end = time.time()
            contract_time[task, size] = time_end - time_start

            print(f'Contracted for {task}-{size} in {contract_time[task, size]:.2f} seconds.')
            n_qubits = len(qig.graph.nodes)
            print(f'QIG size after contraction: {n_qubits} squbits, {len(qig.graph.edges)} edges')

            with open(result_file, 'wb') as f:
                pickle.dump(contract_time, f)


def plot_efficiency(folder = 'result/efficiency/'):
    tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'MCMT' ]
    # tasks = [ 'QFT' ]
    # tasks = [ 'Grover' ]

    # sizes = [ 4, 8, ]
    # sizes = [ 16, 32, 64]
    # sizes = [ 48, 64, 96 ]
    # sizes = [128, 256, 384, 512, 768, 1024]
    # sizes = [128, 256, 384, 512]
    # sizes = [16]
    # sizes = [32]
    # sizes = [ 64,]
    sizes = [ 96, ]
    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    SolverClasses = [ TACOORIG, TACONL, TACOL, ]
    # SolverClasses = [ TACOORIG, TACONL, ]

    markers = ['o', 's', '^', 'x', '*', 'D', 'v', 'P']

    for size in sizes:
        for task in tasks:
            # one image for each task and size
            plt.figure()
            plt.rcParams.update({'font.size': 18})
            plt.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.16)

            for SolverClass, marker in zip(SolverClasses, markers):
                result_file = f'{folder}objs-{task}-{size}-{SolverClass.__name__}.pkl'
                # objs: list of (time, obj)
                objs = pickle.load(open(result_file, 'rb'))
                if objs is None or len(objs) == 0:
                    plt.plot([], label=SolverClass.__name__, marker=marker)
                else:
                    plt.plot([o[0] for o in objs], [o[1] for o in objs], label=SolverClass.__name__, marker=marker)

                print(f'{task}-{size}-{SolverClass.__name__}:')
                print(objs)
            plt.title(f'{task} - {size}')
            plt.xlabel('Time')
            plt.ylabel('Objective Value')
            plt.legend()
            plt.savefig(f'{folder}plot-{task}-{size}.png')
            plt.clf()


def plot_efficiency_contracted(folder = 'result/efficiency/'):
    tasks = [ 'QFT', 'Grover', 'MCMT' ]
    # tasks = [ 'MCMT' ]
    # tasks = [ 'QFT' ]
    # tasks = [ 'Grover' ]

    # sizes = [ 16, 32, 64]
    # sizes = [ 48, 64, 96 ]
    # sizes = [128, 256, 384, 512, 768, 1024]
    # sizes = [512,]
    # sizes = [768, 1024]
    # sizes = [16]
    # sizes = [32]
    sizes = [ 64,]
    # sizes = [ 96, ]
    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    SolverClasses = [ TACOORIG, TACONL, TACOL, ]
    # SolverClasses = [ TACOORIG, TACONL, ]
    # SolverClasses = [ TACOPA, TACOL ]
    names = ['TACOL+EC+PA', 'TACOL+EC']

    markers = ['o', 's', '^', 'x', '*', 'D', 'v', 'P']

    for size in sizes:
        for task in tasks:
            # one image for each task and size
            plt.figure()
            plt.rcParams.update({'font.size': 18})
            plt.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.16)

            for SolverClass, marker, name in zip(SolverClasses, markers, names):
                result_file = f'{folder}objs-{task}-{size}-{SolverClass.__name__}.pkl'
                # objs: list of (time, obj)
                if not os.path.exists(result_file):
                    plt.plot([], label=name, marker=marker)
                else:
                    objs = pickle.load(open(result_file, 'rb'))
                    if objs is None or len(objs) == 0:
                        plt.plot([], label=name, marker=marker)
                    else:
                        plt.plot([o[0] for o in objs], [o[1] for o in objs], label=name, marker=marker)

                    print(f'{task}-{size}-{name}:')
                    print(objs)
            plt.title(f'{task} - {size}')
            plt.xlabel('Time')
            plt.ylabel('Objective Value')
            plt.legend()
            plt.savefig(f'{folder}plot-{task}-{size}.png')
            plt.clf()


def plot_contraction(folder = 'result/efficiency/'):
    result_file = f'{folder}contraction_times.pkl'
    contract_time = {}
    if os.path.exists(result_file):
        with open(result_file, 'rb') as f:
            contract_time = pickle.load(f)

    # print('Contraction times:')
    # print(contract_time)
    tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'MCMT' ]
    # tasks = [ 'QFT' ]
    # tasks = [ 'Grover' ]

    sizes = [192, 256, 384, 512, 768, 1024]

    markers = ['o', 's', '^', 'x', '*', 'D', 'v', 'P']

    # one image for all
    # each task has a line
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.16)

    for task in tasks:
        times = []
        for size in sizes:
            if (task, size) in contract_time:
                times.append(contract_time[task, size])
            else:
                times.append(None)

        plt.plot(sizes, times, label=task, marker='o')
        print(f'Contraction times for {task}:')
        print(times)

    plt.title(f'Contraction Time')
    plt.xlabel('Circuit Size (Number of Qubits)')
    plt.ylabel('Contraction Time (seconds)')
    plt.legend()
    plt.savefig(f'{folder}contraction-time.png')
    plt.clf()


if __name__ == "__main__":


    # run_tests()
    plot_efficiency()

    # run_contraction()
    # plot_contraction()

    # run_tests_contracted()
    # plot_efficiency_contracted()
    


