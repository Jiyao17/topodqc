
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from src.circuit import QIG, QASM_FILES, RandomQIG
from src.solver import TACO, TACOORIG, TACONL, TACOL



def show_cluster(folder = 'result/cluster/'):
    adr4 = 'result/cluster/objs-adr4.pkl'
    clip = 'result/cluster/objs-clip.pkl'
    co14 = 'result/cluster/objs-co14.pkl'
    # merge all into objs
    objs = {}
    for filename in [adr4, clip, co14]:
        objs.update(pickle.load(open(filename, 'rb')))
    # objs = pickle.load(open(filename, 'rb'))
    # timeout = 120
    np.random.seed(0)
    # qigs = [ QIG.from_qasm(filename) for filename in QASM_FILES ]
    # qbits = [24, 32]
    # pairs = [ int(qbit*(3*qbit-3)/8) for qbit in qbits ]
    # print(pairs)
    # print(cnot_nums)

    # rand_qigs = [ 
        # RandomQIG(qbit, pair, (1, 100)) for qbit, pair in zip(qbits, pairs)
    # ]
    # qigs = qigs[:1] + rand_qigs
    # circ_name = "adr4"
    # qigs_names = [ circ_name ]
    qigs_names = [ "adr4", "clip", "co14" ][:3]

    clusters = [
        (8, 2, 4, 8),
        (8, 3, 4, 8),
        (8, 4, 4, 8),
    ]
    cluster_names = [ "small", "medium", "large" ]

    # solvers = [ TACOORIG, TACONL, TACOL ]

    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.1)
    # name = circ_name
    bar_list = [[] for _ in range(len(qigs_names))]
    for i, qig_name in enumerate(qigs_names):
        
        for cluster, cluster_name in zip(clusters, cluster_names):
            
            key = (cluster, qig_name, TACOL)
            if key not in objs:
                continue
            l_objs = objs[key]
            cluster_obj = l_objs[-1][1]

            bar_list[i].append(cluster_obj)

    bars = np.array(bar_list)
    bars = bars.T
    # plot multiple bars
    barWidth = 0.25
    r1 = np.arange(len(bars[0]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, bars[0], width=barWidth, edgecolor='grey', label='small')
    plt.bar(r2, bars[1], width=barWidth, edgecolor='grey', label='medium')
    plt.bar(r3, bars[2], width=barWidth, edgecolor='grey', label='large')

    # plt.xlabel('circuits')
    plt.ylabel('Objective')
    plt.xticks([r + barWidth for r in range(len(bar_list[0]))], ['adr4', 'clip', 'co14'])
    # y log scale
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{folder}cluster.png')
            




if __name__ == "__main__":

    # run_tests_real()
    # run_tests_rand()
    show_cluster()
    # test()
