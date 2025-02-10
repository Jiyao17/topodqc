

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from src.circuit.qig import QIG
from src.solver import TACOORIG, TACONL, TACOL


def plot_real():
    objs_file = 'result/objs-real-adr4.pkl'
    objs = pickle.load(open(objs_file, 'rb'))
    print(objs.items())

    folder = 'result/efficiency/'

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
    cluster = clusters[0]
    solvers = [TACOORIG, TACONL, TACOL]
    sol_names = ['TACO', 'TACO-NL', 'TACO-L']
    markers = ['o', 's', '^']

    # print(objs.keys())

    # cluster (8, 2, 2, 12)
    qig_name = 'adr4'
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.16)
    for solver, name, marker in zip(solvers, sol_names, markers):
        orig_objs = objs[(cluster, qig_name, solver)]
        orig_times = [obj[0] for obj in orig_objs]
        orig_objs = [obj[1] for obj in orig_objs]
        plt.plot(orig_times, orig_objs, label=name, marker=marker)
    plt.xlabel('Time (s)')
    plt.ylabel('Objective')
    # scientific notation y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.title(f'{qig_name} on (8, 2, 2, 12)')
    plt.legend()
    plt.savefig(folder + f'{qig_name}_8_2_4_8.png')
    plt.close()


    # qig_name = 'clip'
    # plt.figure()
    # plt.rcParams.update({'font.size': 18})
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.16)
    # for solver, name, marker in zip(solvers, sol_names, markers):
    #     orig_objs = objs[(cluster, qig_name, solver)]
    #     orig_times = [obj[0] for obj in orig_objs]
    #     orig_objs = [obj[1] for obj in orig_objs]
    #     plt.plot(orig_times, orig_objs, label=name, marker=marker)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Objective')
    # # scientific notation y-axis
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.legend()
    # plt.savefig(folder + f'{qig_name}_8_2_2_8.png')
    # plt.close()



    # qig_name = 'co14'
    # plt.figure()
    # plt.rcParams.update({'font.size': 18})
    # plt.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.16)
    # for solver, name, marker in zip(solvers, sol_names, markers):
    #     orig_objs = objs[(cluster, qig_name, solver)]
    #     orig_times = [obj[0] for obj in orig_objs]
    #     orig_objs = [obj[1] for obj in orig_objs]
    #     plt.plot(orig_times, orig_objs, label=name, marker=marker)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Objective')
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # # plt.title(f'{qig_name} on (8, 4, 4, 16)')
    # plt.legend()
    # plt.savefig(folder + f'{qig_name}_8_2_2_8.png')
    # plt.close()

def plot_rand():
    objs_file = 'result/objs-ml.pkl'
    objs = pickle.load(open(objs_file, 'rb'))
    print(objs.keys())

    folder = 'result/efficiency/'

    # qigs_names = [ "adr4", "clip", "co14" ]
    qigs_names = [ "rand24", "rand32" ]

    clusters = [
        # (4, 4, 4, 4),
        (8, 3, 4, 10),
        # (12, 2, 4, 14),
        # (8, 4, 4, 10),
        (8, 4, 4, 12),
        # (16, 2, 4, 20),
    ]

    solvers = [TACOORIG, TACONL, TACOL]
    sol_names = ['TACO', 'TACO-NL', 'TACO-L']
    markers = ['o', 's', '^']

    # print(objs.keys())

    # cluster (8, 2, 2, 12)
    # qig_name = 'rand16'
    # cluster = clusters[0]
    # plt.figure()
    # plt.rcParams.update({'font.size': 18})
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.16)
    # for solver, name, marker in zip(solvers, sol_names, markers):
    #     orig_objs = objs[(cluster, qig_name, solver)]
    #     orig_times = [obj[0] for obj in orig_objs]
    #     orig_objs = [obj[1] for obj in orig_objs]
    #     plt.plot(orig_times, orig_objs, label=name, marker=marker)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Objective')
    # # scientific notation y-axis
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # # plt.title(f'{qig_name} on (8, 2, 2, 12)')
    # plt.legend()
    # plt.savefig(folder + f'{qig_name}_8_2_4_10.png')
    # plt.close()


    qig_name = 'rand24'
    cluster = clusters[0]
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.16)
    for solver, name, marker in zip(solvers, sol_names, markers):
        
        orig_objs = objs[(cluster, qig_name, solver)]
        if orig_objs is None:
            continue
        orig_times = [obj[0] for obj in orig_objs]
        orig_objs = [obj[1] for obj in orig_objs]
        plt.plot(orig_times, orig_objs, label=name, marker=marker)
    plt.xlabel('Time (s)')
    plt.ylabel('Objective')
    # scientific notation y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.savefig(folder + f'{qig_name}_8_3_4_10.png')
    plt.close()



    qig_name = 'rand32'
    cluster = clusters[1]
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.16)
    for solver, name, marker in zip(solvers, sol_names, markers):
        orig_objs = objs[(cluster, qig_name, solver)]
        orig_times = [obj[0] for obj in orig_objs]
        orig_objs = [obj[1] for obj in orig_objs]
        plt.plot(orig_times, orig_objs, label=name, marker=marker)
    plt.xlabel('Time (s)')
    plt.ylabel('Objective')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.title(f'{qig_name} on (8, 4, 4, 16)')
    plt.legend()
    plt.savefig(folder + f'{qig_name}_8_4_4_12.png')
    plt.close()



if __name__ == "__main__":
    # plot_real()
    plot_rand()