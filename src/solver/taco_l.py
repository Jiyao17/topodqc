


import sys

import gurobipy as gp
import numpy as np

from .taco import TACO
from ..circuit.qig import QIG, RandomQIG
from .type import ProcMemNum



class TACOL(TACO):
    
    def __init__(self, 
            qig: QIG, 
            mems: list[ProcMemNum], 
            comms: list[ProcMemNum], 
            W: int,
            edge_weights: dict[tuple[int, int], float]=None,
            timeout: int = 600
            ) -> None:
        super().__init__(qig, mems, comms, W, timeout)

        self.edge_weights = edge_weights
        if self.edge_weights is not None:
            edge_num = int(len(self.qpus) * (len(self.qpus) - 1) // 2)
            assert len(self.edge_weights) == edge_num, \
                "Edge weights length does not match the number of edges."
        # set NonConvex to 1
        # self.model.setParam('NonConvex', 1)

    def build(self):
        self.add_vars()
        self.add_alloc_constrs()
        self.add_path_constrs()
        self.add_topology_constrs()
        self.set_obj()

    def add_path_constrs(self):
        # each node is allocated to exactly one processor
        self.y = {}
        for i in self.qpus:
            for j in self.qpus:
                if i < j:
                    y_expr = gp.QuadExpr(0)
                    for a in self.squbits:
                        for b in self.squbits:
                            if a < b and self.c[a, b] > 0:
                                y_expr += self.x[a, i] * self.x[b, j] + self.x[b, i] * self.x[a, j]

                    self.y[i, j] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}_{j}')
                    self.model.addConstr(self.y[i, j] <= y_expr, name=f'y_def1_{i}_{j}')
                    # self.model.addConstr(y_expr <= self.y[i, j] * (len(self.squbits) // 2))
                    self.model.addConstr(y_expr <= self.y[i, j] * self.mems[i] * self.mems[j],
                                         name=f'y_def2_{i}_{j}')
                    self.model.addConstr(y_expr <= self.mems[i] * self.mems[j],
                                         name=f'y_expr_{i}_{j}')
                    # self.model.addConstr(y <= self.y[i, j] * len(self.qubits))
                    
                    # for start node i
                    oflow = gp.quicksum(self.p[i, j, i, u] for u in self.qpus if u != i)
                    iflow = gp.quicksum(self.p[i, j, u, i] for u in self.qpus if u != i)
                    self.model.addConstr((oflow - iflow) == self.y[i, j], name=f'flow_constr_start_{i}_{j}')
                    # self.model.addConstr(oflow - iflow == 1)
                    # for end node j
                    oflow = gp.quicksum(self.p[i, j, j, u] for u in self.qpus if u != j)
                    iflow = gp.quicksum(self.p[i, j, u, j] for u in self.qpus if u != j)
                    self.model.addConstr((oflow - iflow) == -self.y[i, j], name=f'flow_constr_end_{i}_{j}')
                    # self.model.addConstr(oflow - iflow == -1)
                    # for intermediate nodes
                    for u in self.qpus:
                        if u != i and u != j:
                            oflow = gp.quicksum(self.p[i, j, u, v] for v in self.qpus if v != u)
                            iflow = gp.quicksum(self.p[i, j, v, u] for v in self.qpus if v != u)
                            # add the cubic term constraint
                            self.model.addConstr((oflow - iflow) == 0, name=f'flow_constr_inter_{i}_{j}_{u}')
                            # self.model.addConstr(oflow - iflow == 0)
                    
        # cancel no demand path
        for i in self.qpus:
            for j in self.qpus:
                if i < j:
                    usage = 0
                    for u in self.qpus:
                        for v in self.qpus:
                            if u < v:
                                usage += self.p[i, j, u, v] + self.p[i, j, v, u]
                                # self.model.addConstr(self.p[i, j, u, v] <= self.y[i, j])
                                # self.model.addConstr(self.p[i, j, v, u] <= self.y[i, j])
                                # self.model.addConstr(self.p[i, j, u, v] + self.p[i, j, v, u] <= self.y[i, j])
                    self.model.addConstr(usage <= self.y[i, j] * (len(self.qpus) // 2) , name=f'path_usage_{i}_{j}')

    def add_topology_constrs(self):
        self.w = {}
        total = 0
        for u in self.qpus:
            for v in self.qpus:
                if u < v:
                    self.w[u, v] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'w_{u}_{v}')
                    w = 0
                    for i in self.qpus:
                        for j in self.qpus:
                            if i < j:
                                w += self.p[i, j, u, v] + self.p[i, j, v, u]
                                
                    self.model.addConstr(self.w[u, v] <= w, name=f'edge_usage1_{u}_{v}')
                    self.model.addConstr(w <= self.w[u, v] * len(self.qpus) * (len(self.qpus) - 1), name=f'edge_usage2_{u}_{v}')
                    # self.model.addConstr(w <= self.w[u, v] * len(self.procs) * len(self.procs))
                    total += self.w[u, v] 

        self.model.addConstr(total <= self.W, name='total_edge_usage')

        # if candidate graph is not complete, add comms constraints
        # if self.edge_weights is not None:
        #     for (i, j), weight in self.edge_weights.items():
        #         if weight >= float('inf'):
        #             # no edge
        #             self.model.addConstr(self.w[i, j] == 0, name=f'no_edge_{i}_{j}')

        for u in self.qpus:
            adj = 0
            for z in self.qpus:
                if u != z:
                    if u < z:
                        adj += self.w[u, z]
                    else:
                        adj += self.w[z, u]
            self.model.addConstr(adj <= self.comms[u], name=f'comm_{u}')
                    
    def set_obj(self):
        total = 0
        demand_ub = sum(self.c[a, b] for a in self.squbits for b in self.squbits if a < b and self.c[a, b] > 0)
        for i in self.qpus:
            for j in self.qpus:
                if i < j:
                    path_weight = 0
                    for u in self.qpus:
                        for v in self.qpus:
                            if u < v:
                                weight = self.edge_weights[u, v] if self.edge_weights is not None else 1
                                path_weight += (self.p[i, j, u, v] + self.p[i, j, v, u]) * weight
                    
                    _demand = 0
                    for a in self.squbits:
                        for b in self.squbits:
                            if a < b and self.c[a, b] > 0:
                                _demand += self.c[a, b] * self.x[a, i] * self.x[b, j] \
                                            + self.c[a, b] * self.x[b, i] * self.x[a, j]

                    demand = self.model.addVar(vtype=gp.GRB.INTEGER, name=f'demand_{i}_{j}', lb=0, ub=demand_ub)
                    self.model.addConstr(demand == _demand, name=f'demand_constr_{i}_{j}')
                    # self.model.addConstr(_demand >= demand)

                    total += path_weight * demand

        self.model.setObjective(total, gp.GRB.MINIMIZE)

    # def get_topology(self):
        
    #     edges = []
    #     for u in self.qpus:
    #         for v in self.qpus:
    #             if u < v:
    #                 if self.w[u, v].x > 0.5:
    #                     edges.append((u, v))
    #     return edges

if __name__ == "__main__":
    np.random.seed(0)
    proc_num = 2
    mem = 4
    comm = 1

    qubit_num = proc_num * mem
    # demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = int(qubit_num * (qubit_num-1) / 6) # max
    # demand_pair = qubit_num * 2 # moderate

    # qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # print(sorted(qig.demands))
    # qig = QIG.from_qasm('src/circuit/src/mcmt_2c_2t.qasm')
    qig = QIG.from_qasm('src/circuit/src/grover_8.qasm')
    # qig.contract(4, inplace=True)

    # 256 homogeneous
    mems = [mem, ] * proc_num
    comms = [comm, ] * proc_num

    # mems = [64, 64, 32, 32, 16, 16, 16, 16]
    # comms = [16, 16, 8, 8, 4, 4, 4, 4]
    # mems = [32, 32, 16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 8, 8, 4, 4, 4, 4]
    # mems = [16, 16, 8, 8, 8, 8]
    # comms = [8, 8, 4, 4, 4, 4]
    # mems = [8, 8, 4, 4, 4, 4]
    # comms = [4, 4, 4, 4, 4, 4]
    W = int(proc_num * (proc_num-1) / 2)
    # W = 1e6
    # W = (proc_num - 1)
    # W = proc_num  + 1

    # qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)

    from src.solver.qboson.sa import SASolver, TACOSA

    sa = SASolver(process_num=16)
    tacol = TACOL(qig, mems, comms, W,)
    tacol.build()
    tacol.model.update()
    # results = sa.solve(tacol.model, slack_bound=max(proc_num, mem*mem), timeout=600)
    results = sa.solve(tacol.model, slack_bound=3, timeout=300)
    # tacosa = TACOSA(qig, mems, comms, W,)
    # tacosa.build()
    # tacosa.solve()

    # tacol = TACOL(qig, mems, comms, W,)
    # tacol.build()
    # tacol.model.update()

    # print("var num:", tacol.model.numVars)
    # print("constr num:", tacol.model.numConstrs)
    # print(tacol.solve())

    # set all edge variables w to the solution
    # for u in tacol.qpus:
    #     for v in tacol.qpus:
    #         if u < v:
    #             # print((u, v), tacol.w[u, v].x)
    #             if tacol.w[u, v].x > 0.5:
    #                 tacol.model.addConstr(tacol.w[u, v] == 1, name=f'fix_w_{u}_{v}')
    #             else:
    #                 tacol.model.addConstr(tacol.w[u, v] == 0, name=f'fix_w_{u}_{v}')
    # tacol.model.update()

    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)

    # for a in model.qubits:
    #     for i in model.procs:
    #         print((a, i), model.x[a, i].x)
    
    # path_lengths = model.get_results()
    # print(path_lengths)

    # check var types
    # for v in tacol.model.getVars():
    #     assert v.VType in [gp.GRB.BINARY], f"Var {v.VarName} has type {v.VType}, expected BINARY"
    # sa = SASolver(process_num=1)
    # slack_bound = max(proc_num, mem)
    # best_vals = sa.solve(
    #     tacol.model, 
    #     slack_bound=slack_bound, 
    #     timeout=600
    #     )
    # print("Best values over time (time, obj, violation):")
    # print(best_vals)

