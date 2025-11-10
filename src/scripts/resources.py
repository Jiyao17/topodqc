

# performance on different setting of parameters
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


def run_tests_resources(folder = 'result/efficiency/'):
    timeout = 300
    np.random.seed(42)

    tasks = [ 'MCMT', 'QFT', 'Grover', ]
    # tasks = [ 'MCMT' ]
    # tasks = [ 'QFT', 'Grover' ]
    # tasks = [ 'Grover' ]

    # sizes = [ 32, 48, 64, 96 ]
    # sizes = [ 24, 32 ]
    sizes = [ 64, 96, 128, ]
    # sizes = [ 16 ]
    # sizes = [ 24 ]
    # sizes = [ 32 ]
    # sizes = [ 192, ]
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
        96: ([12,]*8, [3,]*8, 10),
        128: ([16,]*8, [3,]*8, 10),
        192: ([16,]*8 + [8,]*8, [4,]*8 + [3,]*8, 20),
        256: ([16,]*16, [4,]*16, 30),
        384: ([16,]*16 + [8,]*16, [4,]*16 + [3,]*16, 60),
        512: ([32,]*16, [8,]*16, 30),
        768: ([32,]*16 + [16,]*16, [8,]*16 + [4,]*16, 60),
        1024: ([32,]*32, [8,]*32, 60),
    }




    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    # SolverClasses = [ TACOL, TACOORIG, TACONL,  ]
    # SolverClasses = [ TACOPA, TACOL ]
    SolverClasses = [ TACOPA ]
    # SolverClasses = [ TACOSA ]

    for task in tasks:
        for size in sizes:
            qasm_file = QASM_FILES[task, 64]
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

                with open(f'{folder}objs-{task}-64-{size}-{SolverClass.__name__}.pkl', 'wb') as f:
                    pickle.dump(objs, f)


                print(f'Simulations for {task}-{size} with {SolverClass.__name__} done!')

                # print("Objs:", objs)



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
    sizes = [ 64, 96, 128 ]
    # sizes = [ 96, ]
    # SolverClasses = [ TACOORIG, TACONL, TACOL, TACOSA ]
    # SolverClasses = [ TACOORIG, TACONL, TACOL, ]
    # SolverClasses = [ TACOORIG, TACONL, ]
    SolverClasses = [ TACOPA ]
    names = ['TACOL+EC+PA',]

    markers = ['o', 's', '^', 'x', '*', 'D', 'v', 'P']

    mat = np.zeros((len(sizes), len(tasks), )) + np.nan
    for size in sizes:
        for task in tasks:
            # one image for each task and size
            plt.figure()
            plt.rcParams.update({'font.size': 18})
            plt.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.16)

            for SolverClass, marker, name in zip(SolverClasses, markers, names):
                result_file = f'{folder}objs-{task}-64-{size}-{SolverClass.__name__}.pkl'
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

                    mat[sizes.index(size), tasks.index(task)] = objs[-1][1]

    # plot grouped bar chart using the collected matrix `mat`
    # mat shape: (n_sizes, n_tasks)
    n_tasks = len(tasks)
    n_sizes = len(sizes)

    # Use a narrower figure suitable for IEEE double-column layouts
    # slightly narrower figure; we'll tighten spacing between task groups below
    plt.figure(figsize=(5.5, 3.5))
    plt.rcParams.update({'font.size': 12})

    # reduce the space between task groups by using a group_gap < 1.0
    group_gap = 0.6
    index = np.arange(n_tasks) * group_gap
    # choose a smaller bar width so bars within a group don't overlap
    bar_width = 0.15

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_sizes)]

    for s_idx in range(n_sizes):
        positions = index + (s_idx - (n_sizes - 1) / 2) * bar_width
        vals = mat[s_idx, :]
        # vals may contain nan for missing results; matplotlib will leave gaps
        plt.bar(positions, vals, bar_width, label=f'size {sizes[s_idx]}', color=colors[s_idx], edgecolor='black')

    plt.xlabel('Task')
    plt.ylabel('Final objective value')
    plt.title('Final solver objective by task and resource size')
    plt.xticks(index, tasks)
    plt.legend(title='Sizes')
    plt.tight_layout()

    # tighten x-limits so there's less empty space at the sides
    group_width = n_sizes * bar_width
    plt.xlim(index[0] - group_width, index[-1] + group_width)

    os.makedirs(folder, exist_ok=True)
    out_file = os.path.join(folder, 'comp_resources.png')
    # save with tight bbox and high DPI for publication-quality embedding
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f'Grouped bar plot saved to {out_file}')
    plt.show()



if __name__ == '__main__':
    run_tests_resources()
    # plot_efficiency_resources()

    plot_efficiency_contracted()