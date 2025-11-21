
"""
Application of TACOPA to solve
1. quantum data center networks (Top on Rack switched networks) (SwitchQN)
2. Argonne National Lab network (ANL) in Chicago area (SeQUeNCe)

"""



from collections import defaultdict
import time

import numpy as np
import os

from src.circuit.qig import QIG, RandomQIG
from src.solver.taco_pa import TACOPA
from src.solver.taco_l import TACOL
from src.utils.graph import draw_topology


def form_racks(
        mems: list[int], rack_size: int, 
        in_rack_cost: float, cross_rack_cost: float
        ) -> dict[tuple[int, int], float]:
    
    # map from mem to list of proc ids
    qpus = defaultdict(list)
    for i, mem in enumerate(mems):
        qpus[mem].append(i)

    # put only processors with same memory size in the same rack
    # form each rack greedily
    # remaining processors are put in the last rack
    racks = []
    # from proc id to rack id
    get_rack = {}
    for mem, procs in qpus.items():
        rack_num = len(procs) // rack_size
        for r in range(rack_num):
            rack_procs = procs[r*rack_size:(r+1)*rack_size]
            racks.append(rack_procs)
            for p in rack_procs:
                get_rack[p] = len(racks) - 1
        # remaining procs
        rem_procs = procs[rack_num*rack_size:]
        if len(rem_procs) > 0:
            racks.append(rem_procs)
            for p in rem_procs:
                get_rack[p] = len(racks) - 1

    # form edge weights
    edge_weights = {}
    for i in range(len(mems)):
        for j in range(i+1, len(mems)):
            if get_rack[i] == get_rack[j]:
                edge_weights[(i, j)] = in_rack_cost
            else:
                edge_weights[(i, j)] = cross_rack_cost

    return edge_weights


def form_ANL() -> dict[tuple[int, int], float]:
    """
    Form the ANL topology
    """
    sites = {
        0: 'NU-Evanston',
        1: 'StarLight',
        2: 'UChicago-PME',
        3: 'UChicago-HC',
        4: 'Fermilab-1',
        5: 'Fermilab-2',
        6: 'Argonne-1',
        7: 'Argonne-2',
        8: 'Argonne-3',
    }

    distances = {
        (0, 1): 20,
        (1, 2): 16,
        (1, 4): 66,
        (1, 6): 54,
        (2, 3): 2,
        (2, 6): 42,
        (4, 5): 2,
        (4, 6): 53,
        (6, 7): 0.1,
        (6, 8): 41.8,
    }
    
    
    edge_weights = {}
    for i in range(len(sites)):
        for j in range(i+1, len(sites)):
            if (i, j) in distances:
                edge_weights[(i, j)] = pow(2, distances[(i, j)])
            else:
                # max value for non-existing links
                # edge_weights[(i, j)] = sys.float_info.max
                edge_weights[(i, j)] = 1e20

    return edge_weights


def form_ANL_linear_comm() -> dict[tuple[int, int], float]:
    """
    Form the ANL topology
    """
    sites = {
        0: 'NU-Evanston',
        1: 'StarLight',
        2: 'UChicago-PME',
        3: 'UChicago-HC',
        4: 'Fermilab-1',
        5: 'Fermilab-2',
        6: 'Argonne-1',
        7: 'Argonne-2',
        8: 'Argonne-3',
    }

    distances = {
        (0, 1): 20,
        (1, 2): 16,
        (1, 4): 66,
        (1, 6): 54,
        (2, 3): 2,
        (2, 6): 42,
        (4, 5): 2,
        (4, 6): 53,
        (6, 7): 0.1,
        (6, 8): 41.8,
    }
    
    
    edge_weights = {}
    for i in range(len(sites)):
        for j in range(i+1, len(sites)):
            if (i, j) in distances:
                edge_weights[(i, j)] = distances[(i, j)]
            else:
                # max value for non-existing links
                # edge_weights[(i, j)] = sys.float_info.max
                edge_weights[(i, j)] = 1e5

    return edge_weights




def test_SwitchQN():
    np.random.seed(0)
    
    qig = QIG.from_qasm('src/circuit/src/qft_512.qasm')
    # qig = QIG.from_qasm('src/circuit/src/grover_512.qasm')
    # qig = QIG.from_qasm('src/circuit/src/mcmt_256c_256t.qasm')
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
    mems = [32, ] * 25

    comm = 4
    comms = [comm, ] * 25
    W = 1000

    # objs = [31153392.0, 24594416.0, 24577060.0, 24569824.0, 24566264.0]


    # mems = [32, 32, 16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 8, 8, 4, 4, 4, 4]
    # mems = [16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 4, 4, 4, 4]
    # mems = [2, ] * 8
    # comms = [8, ] * 8
    # W = 160
    # W = int(proc_num * (proc_num-1) / 2)
    # W = 1e6
    # W = (proc_num - 1)
    # W = proc_num  + 1

    edge_weights = form_racks(
        mems, rack_size=5, 
        in_rack_cost=1, cross_rack_cost=100*8
        )
    print(edge_weights)

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    model = TACOPA(qig, mems, comms, W, edge_weights)
    # model = TACOPA(qig, mems, comms, W)
    model.build()
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    print("Solving the model...")
    start_time = time.time()
    print(model.solve())
    print(f"Model solved in {time.time() - start_time:.2f} seconds.")

    edges = model.get_topology()
    # print(edges)
    draw_topology(edges, filename=f'result/taco_pa_topology_switchqn-{comm}.png')
    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)


def plot_switchqn_objs():
    interface_nums = [2, 3, 4, 5, 6]

    # qft_1024
    objs = [31153392.0, 24594416.0, 24577060.0, 24569824.0, 24566264.0]
    # grover_1024
    objs = [96426216.0, 75760328.0, 75708136.0]
    # mcmt_512c_512t
    objs = [981687263.3635908, 775808848.0, 775395412.0]

    # 21% 0.12% 

    import matplotlib.pyplot as plt

    plt.plot(interface_nums, objs)
    plt.xlabel('Interface Number')
    plt.ylabel('Objective Value')
    # log y axis
    # plt.yscale('log')
    # plt.title('SwitchQN Objective Values')
    # plt.show()
    plt.savefig('result/switchqn_objs.png')


def plot_switchqn_objs_bars():
    interface_nums = [2, 3, 4,]

    # qft_1024
    # objs_qft = [31153392.0, 24594416.0, 24577060.0, 24569824.0, 24566264.0]
    objs_qft = [31153392.0, 24594416.0, 24577060.0, ]

    # grover_1024
    objs_grover = [96426216.0, 75760328.0, 75708136.0]
    # mcmt_512c_512t
    objs_mcmt = [981687263.3635908, 775808848.0, 775395412.0]


    import matplotlib.pyplot as plt

    # convert to numpy arrays
    a = np.array(objs_qft)
    b = np.array(objs_grover)
    c = np.array(objs_mcmt)

    n = len(interface_nums)
    ind = np.arange(n)
    width = 0.22

    # create a broken y-axis to omit a middle range where MCMT dominates
    lower_max = 0.1e9
    upper_min = 0.7e9
    overall_max = max(a.max(), b.max(), c.max())
    upper_max = max(overall_max * 1.05, upper_min * 1.1)

    fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True,
                                             gridspec_kw={'height_ratios': [2, 1]},
                                             figsize=(5.5, 4))

    # plot all three algorithms on both axes (so MCMT bars are broken)
    ax_lower.bar(ind - width, a, width, label='_nolegend_', color='#4C72B0', edgecolor='black')
    ax_lower.bar(ind, b, width, label='_nolegend_', color='#DD8452', edgecolor='black')
    ax_lower.bar(ind + width, c, width, label='_nolegend_', color='#55A868', edgecolor='black')

    ax_upper.bar(ind - width, a, width, label='QFT', color='#4C72B0', edgecolor='black')
    ax_upper.bar(ind, b, width, label='Grover', color='#DD8452', edgecolor='black')
    ax_upper.bar(ind + width, c, width, label='MCMT', color='#55A868', edgecolor='black')

    # set both axes to log scale and set limits to omit the middle range
    min_val = min(a.min(), b.min(), c.min())
    lower_min = max(min_val * 0.8, 1.0)
    ax_lower.set_yscale('log')
    ax_upper.set_yscale('log')
    ax_lower.set_ylim(lower_min, lower_max)
    ax_upper.set_ylim(upper_min, upper_max)

    # hide the spines between axes
    ax_upper.spines['bottom'].set_visible(False)
    ax_lower.spines['top'].set_visible(False)
    # hide x ticks/labels for the upper part to avoid duplication
    ax_upper.tick_params(labeltop=False, labelbottom=False, bottom=False)
    ax_upper.set_xticks([])

    # diagonal lines to indicate the break
    d = .015  # size of diagonal lines in axes coordinates
    kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
    ax_upper.plot((-d, +d), (-d, +d), **kwargs)
    ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_lower.transAxes, color='k', clip_on=False)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_lower.set_xlabel('Interface Number')
    ax_upper.set_ylabel('Objective Value')
    ax_lower.set_xticks(ind)
    ax_lower.set_xticklabels(interface_nums)

    # show legend only on the upper axis
    ax_upper.legend(ncols=1, fontsize='large')



    plt.tight_layout()

    os.makedirs('result', exist_ok=True)
    out_file = 'result/switchqn_objs_bars_broken.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f'Saved broken-axis bar plot to {out_file}')
    plt.show()


def plot_switchqn_objs_bars_5():
    interface_nums = [2, 3, 4, 5 ]

    # qft_1024
    # objs_qft = [31153392.0, 24594416.0, 24577060.0, 24569824.0, 24566264.0]
    objs_qft = [916560.0, 916560.0 ]

    # grover_1024
    objs_grover = [32224504.0, 29046224.0, ]
    # mcmt_256c_256t
    objs_mcmt = [174436438.0, 64658246.0, 64613226.0, 64597566.0 ]


    import matplotlib.pyplot as plt

    # convert to numpy arrays
    a = np.array(objs_qft)
    b = np.array(objs_grover)
    c = np.array(objs_mcmt)

    n = len(interface_nums)
    ind = np.arange(n)
    width = 0.22

    # create a broken y-axis to omit a middle range where MCMT dominates
    lower_max = 0.1e9
    upper_min = 0.7e9
    overall_max = max(a.max(), b.max(), c.max())
    upper_max = max(overall_max * 1.05, upper_min * 1.1)

    fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True,
                                             gridspec_kw={'height_ratios': [2, 1]},
                                             figsize=(5.5, 4))

    # plot all three algorithms on both axes (so MCMT bars are broken)
    ax_lower.bar(ind - width, a, width, label='_nolegend_', color='#4C72B0', edgecolor='black')
    ax_lower.bar(ind, b, width, label='_nolegend_', color='#DD8452', edgecolor='black')
    ax_lower.bar(ind + width, c, width, label='_nolegend_', color='#55A868', edgecolor='black')

    ax_upper.bar(ind - width, a, width, label='QFT', color='#4C72B0', edgecolor='black')
    ax_upper.bar(ind, b, width, label='Grover', color='#DD8452', edgecolor='black')
    ax_upper.bar(ind + width, c, width, label='MCMT', color='#55A868', edgecolor='black')

    # set both axes to log scale and set limits to omit the middle range
    min_val = min(a.min(), b.min(), c.min())
    lower_min = max(min_val * 0.8, 1.0)
    ax_lower.set_yscale('log')
    ax_upper.set_yscale('log')
    ax_lower.set_ylim(lower_min, lower_max)
    ax_upper.set_ylim(upper_min, upper_max)

    # hide the spines between axes
    ax_upper.spines['bottom'].set_visible(False)
    ax_lower.spines['top'].set_visible(False)
    # hide x ticks/labels for the upper part to avoid duplication
    ax_upper.tick_params(labeltop=False, labelbottom=False, bottom=False)
    ax_upper.set_xticks([])

    # diagonal lines to indicate the break
    d = .015  # size of diagonal lines in axes coordinates
    kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
    ax_upper.plot((-d, +d), (-d, +d), **kwargs)
    ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_lower.transAxes, color='k', clip_on=False)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_lower.set_xlabel('Interface Number')
    ax_upper.set_ylabel('Objective Value')
    ax_lower.set_xticks(ind)
    ax_lower.set_xticklabels(interface_nums)

    # show legend only on the upper axis
    ax_upper.legend(ncols=1, fontsize='large')



    plt.tight_layout()

    os.makedirs('result', exist_ok=True)
    out_file = 'result/switchqn_objs_bars_broken.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    print(f'Saved broken-axis bar plot to {out_file}')
    plt.show()




def test_ANL():
    # np.random.seed(0)

    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig = QIG.from_qasm('src/circuit/src/grover_256.qasm')
    # qig = QIG.from_qasm('src/circuit/src/mcmt_128c_128t.qasm')
    # qig = QIG.from_qasm('src/circuit/src/qft_1024.qasm')
    # qig = QIG.from_qasm('src/circuit/src/qft_1536.qasm')
    qig = QIG.from_qasm('src/circuit/src/grover_1536.qasm')
    # qig = QIG.from_qasm('src/circuit/src/qft_2048.qasm')

    # qig.contract(4, inplace=True)


    mems = [256, ] * 9
    comms = [16, ] * 9
    W = 100

    edge_weights = form_ANL()
    print(edge_weights)

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    # model = TACOPA(qig, mems, comms, W, edge_weights)
    model = TACOL(qig, mems, comms, W, edge_weights)
    # model = TACOPA(qig, mems, comms, W)
    model.build()
    # model.model.addConstr(model.w[6, 7] == 1, name='fix_link_6_7')
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    print("Solving the model...")
    start_time = time.time()
    print(model.solve())
    print(f"Model solved in {time.time() - start_time:.2f} seconds.")

    edges = model.get_topology()
    # print(edges)
    draw_topology(edges, filename='result/taco_pa_topology.png')
    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)



def test_ANL_computation():

    # qig = QIG.from_qasm('src/circuit/src/qft_1536.qasm')
    qig = QIG.from_qasm('src/circuit/src/mcmt_768c_768t.qasm')



    # mems = [256, 256, 256, 64, 256, 64, 256, 64, 64 ] 
    mems = [256, 256, 256, 128, 256, 128, 256, 128, 128 ] 

    comms = [16, ] * 9
    W = 100

    # edge_weights = form_ANL_linear_comm()
    edge_weights = form_ANL()
    print(edge_weights)

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    # model = TACOPA(qig, mems, comms, W, edge_weights)
    model = TACOL(qig, mems, comms, W, edge_weights)
    # model = TACOPA(qig, mems, comms, W)
    model.build()
    # model.model.addConstr(model.w[6, 7] == 1, name='fix_link_6_7')
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    print("Solving the model...")
    start_time = time.time()
    print(model.solve())
    print(f"Model solved in {time.time() - start_time:.2f} seconds.")

    edges = model.get_topology()
    # print(edges)
    draw_topology(edges, filename='result/taco_pa_topology_computation.png')
    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)


def test_ANL_comm():

    # qig = QIG.from_qasm('src/circuit/src/qft_2048.qasm')
    # qig = QIG.from_qasm('src/circuit/src/grover_2048.qasm')
    qig = QIG.from_qasm('src/circuit/src/mcmt_768c_768t.qasm')


    mems = [256, ] * 9
    comms = [16, ] * 9
    W = 100

    edge_weights = form_ANL_linear_comm()
    print(edge_weights)

    print("Pre-processing the graph by contraction...")
    start_time = time.time()
    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)'
    print(f"Contraction done in {time.time() - start_time:.2f} seconds.")

    print("Building the model...")
    start_time = time.time()
    # model = TACOPA(qig, mems, comms, W, edge_weights)
    model = TACOL(qig, mems, comms, W, edge_weights)
    # model = TACOPA(qig, mems, comms, W)
    model.build()
    # model.model.addConstr(model.w[6, 7] == 1, name='fix_link_6_7')
    print(f"Model built in {time.time() - start_time:.2f} seconds.")
    print("Solving the model...")
    start_time = time.time()
    print(model.solve())
    print(f"Model solved in {time.time() - start_time:.2f} seconds.")

    edges = model.get_topology()
    # print(edges)
    draw_topology(edges, filename='result/taco_pa_topology_comm.png')
    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)



if __name__ == "__main__":
    # test_ANL()
    # test_ANL_computation()
    # test_ANL_comm()
    test_SwitchQN()
    # plot_switchqn_objs()
    # plot_switchqn_objs_bars()