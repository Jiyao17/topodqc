
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


def run_tests_contracted(folder = 'result/efficiency/'):
    timeout = 300
    np.random.seed(42)

    tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'MCMT' ]
    # tasks = [ 'QFT', 'Grover' ]
    # tasks = [ 'Grover' ]

    # sizes = [ 32, 48, 64, 96 ]
    # sizes = [ 24, 32 ]
    # sizes = [ 64, 96 ]
    # sizes = [ 16 ]
    # sizes = [ 24 ]
    # sizes = [ 32 ]
    sizes = [ 192, ]
    # sizes = [ 256, 384, 512, 768, 1024 ]
    # sizes = [ 384, ]
    # sizes = [ 512, ]
    # sizes = [ 768, ]
    # sizes = [ 1024 ]



    clusters = {
        16: ([4,]*4, [3,]*4, 5),
        24: ([4,]*6, [3,]*6, 8),
        32: ([8,]*4, [3,]*4, 5),
        48: ([8,]*4 + [4,]*4, [3,]*8, 10),
        64: ([8,]*8, [3,]*8, 10),
        96: ([8,]*8 + [4,]*8, [3,]*16, 20),
        192: ([16,]*8 + [8,]*8, [4,]*8 + [3,]*8, 20),
        256: ([16,]*16, [4,]*16, 30),
        384: ([16,]*16 + [8,]*16, [4,]*16 + [3,]*16, 60),
        512: ([32,]*16, [8,]*16, 30),
        768: ([32,]*16 + [16,]*16, [8,]*16 + [4,]*16, 60),
        1024: ([32,]*32, [8,]*32, 60),
    }




    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    # SolverClasses = [ TACOL, TACOORIG, TACONL,  ]
    SolverClasses = [ TACOPA, TACOL ]
    # SolverClasses = [ TACOPA ]
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




def plot_efficiency_contracted(folder = 'result/efficiency/'):
    # tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'MCMT' ]
    # tasks = [ 'QFT' ]
    # tasks = [ 'Grover' ]
    # task = 'QFT'
    # task = 'Grover'
    task = 'MCMT'

    sizes = [192, ]
    # sizes = [ 16, 32, 64, 96]
    # sizes = [ 48, 64, 96 ]
    # sizes = [256, 384, 512, 768, 1024]
    # sizes = [256, 512, 1024]

    SolverClasses = [ TACOL, TACOPA ]
    names = ['EC', 'EC+PA']

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'x', '*', 'D', 'v', 'P']

    # plot one image, four lines
    # two lines for each solver: objective value and time
    # for each solver, solid line for objective value, dashed line for time
    # the x axis is the total number of qubits (in sizes)
    # two y axes: left for best objective value, right for time taken to first time get the best objective value
    plt.figure(figsize=(10, 6))

    objs = { name: [] for name in names }
    times = { name: [] for name in names }
    for size in sizes:
        tacol_file = f'{folder}objs-{task}-{size}-TACOL.pkl'
        tacopa_file = f'{folder}objs-{task}-{size}-TACOPA.pkl'
        tacol_objs = None
        tacopa_objs = None
        if os.path.exists(tacol_file):
            with open(tacol_file, 'rb') as f:
                tacol_objs = pickle.load(f)
            if tacol_objs is not None:
                objs['EC'].append(tacol_objs[-1][1])
                for t, obj in tacol_objs:
                    if obj <= tacol_objs[-1][1]:
                        times['EC'].append(t)
                        break
            else:
                objs['EC'].append(None)
                times['EC'].append(None)
        else:
            objs['EC'].append(None)
            times['EC'].append(None)

        if os.path.exists(tacopa_file):
            with open(tacopa_file, 'rb') as f:
                tacopa_objs = pickle.load(f)
            if tacopa_objs is not None:
                objs['EC+PA'].append(tacopa_objs[-1][1])
                for t, obj in tacopa_objs:
                    if obj <= tacopa_objs[-1][1]:
                        times['EC+PA'].append(t)
                        break
            else:
                objs['EC+PA'].append(None)
                times['EC+PA'].append(None)
        else:
            objs['EC+PA'].append(None)
            times['EC+PA'].append(None)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for i, name in enumerate(names):
        ax1.plot(sizes, objs[name], marker=markers[i], label=f'Obj - {name}', color=colors[i])
        ax2.plot(sizes, times[name], marker=markers[i], linestyle='--', label=f'Time - {name}', color=colors[i])
    
    print("Objs:", objs)
    print("Times:", times)
    
    # log scale for value axis
    ax1.set_yscale('log')
    # ax2.set_yscale('log')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Best Objective Value')
    ax2.set_ylabel('Time to Best Objective Value (s)')
    ax1.set_title(f'Solver Efficiency on Contracted {task} Circuits')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{folder}plot_efficiency_contracted_{task}.png')
    # plt.show()  



if __name__ == "__main__":


    # run_contraction()
    # plot_contraction()

    # plot_efficiency()

    # run_tests_contracted()
    plot_efficiency_contracted()
    


