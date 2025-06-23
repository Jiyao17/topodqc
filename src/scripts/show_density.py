

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from src.circuit.qig import QIG
from src.solver import TACOORIG, TACONL, TACOL


if __name__ == "__main__":
    adr4_file = 'result/density/objs-adr4.pkl'
    rand24_file = 'result/density/objs-rand24.pkl'
    rand32_file = 'result/density/objs-rand32.pkl'
    
    adr4_objs = pickle.load(open(adr4_file, 'rb'))
    rand24_objs = pickle.load(open(rand24_file, 'rb'))
    rand32_objs = pickle.load(open(rand32_file, 'rb'))

    objs_list = [adr4_objs, rand24_objs, rand32_objs]
    # objs_list = [adr4_objs, rand24_objs]

    folder = 'result/density/'

    # qigs_names = [ "adr4", "clip", "co14", "rand16", "rand24", "rand32" ]
    qigs_names = [ "adr4", "rand24", "rand32" ]
    # qigs_names = [ "adr4", "rand24", ]
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
    # solvers = [TACOORIG, TACONL, TACOL]
    # sol_names = ['TACO', 'TACO-NL', 'TACO-L']


    # plot multiple bar plot
    bars_list = []
    for i, qig_name in enumerate(qigs_names):
        cluster = clusters[i]
        objs = objs_list[i]
        bars = []
        for c in cluster:
            n_qpu, n_mem, n_comm, W = c
            mems, comms = [n_mem]*n_qpu, [n_comm]*n_qpu
            capacity = n_qpu * n_mem
            
            objs_c = objs[(c, qig_name, TACOL)][-1][1]
            # adr4_objs_c = adr4_objs[(c, qig_name, TACOL)]
            # rand24_objs_c = rand24_objs[(c, qig_name, TACOL)]
            # rand32_objs_c = rand32_objs[(c, qig_name, TACOL)]
            bars.append(objs_c)
            # bars.append(rand24_objs_c)

        bars_list.append(bars)

    bars_list = np.array(bars_list)
    bars_list = bars_list.T




    # set width of bar
    barWidth = 0.25
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # adjust boarder
    plt.subplots_adjust(left=0.14, right=0.94, top=0.94, bottom=0.1)
    # fontsize = 18
    plt.rcParams.update({'font.size': 18})
    # Set position of bar on X axis
    adr4_bars = bars_list[0]
    rand24_bars = bars_list[1]
    rand32_bars = bars_list[2]
    print(adr4_bars)
    print(rand24_bars)
    r1 = np.arange(len(adr4_bars))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    ax.bar(r1, adr4_bars, width=barWidth, edgecolor='grey', label='W=8')
    ax.bar(r2, rand24_bars, width=barWidth, edgecolor='grey', label='W=10')
    ax.bar(r3, rand32_bars, width=barWidth, edgecolor='grey', label='W=12')

    # Add xticks on the middle of the group bars
    # ax.set_xlabel('Circuits')
    ax.set_ylabel('Objective', fontsize=18)
    ax.set_xticks([r + barWidth for r in range(len(adr4_bars))])
    ax.set_xticklabels(['adr4', 'rand24', 'rand32'], fontsize=18)

    # log y-axis
    ax.set_yscale('log')
    # y tick size
    ax.yaxis.set_tick_params(labelsize=18)

    ax.legend()

    plt.savefig(folder + 'density.png')
    plt.close()
