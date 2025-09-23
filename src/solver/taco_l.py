


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
            timeout: int = 600
            ) -> None:
        super().__init__(qig, mems, comms, W, timeout)
        
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
                    self.model.addConstr(self.y[i, j] <= y_expr)
                    # self.model.addConstr(y_expr <= self.y[i, j] * (len(self.squbits) // 2))
                    self.model.addConstr(y_expr <= self.y[i, j] * (len(self.squbits) * (len(self.squbits) - 1)))

                    # self.model.addConstr(y <= self.y[i, j] * len(self.qubits))
                    
                    # for start node i
                    oflow = gp.quicksum(self.p[i, j, i, u] for u in self.qpus if u != i)
                    iflow = gp.quicksum(self.p[i, j, u, i] for u in self.qpus if u != i)
                    self.model.addConstr((oflow - iflow) == self.y[i, j])
                    # self.model.addConstr(oflow - iflow == 1)
                    # for end node j
                    oflow = gp.quicksum(self.p[i, j, j, u] for u in self.qpus if u != j)
                    iflow = gp.quicksum(self.p[i, j, u, j] for u in self.qpus if u != j)
                    self.model.addConstr((oflow - iflow) == -self.y[i, j])
                    # self.model.addConstr(oflow - iflow == -1)
                    # for intermediate nodes
                    for u in self.qpus:
                        if u != i and u != j:
                            oflow = gp.quicksum(self.p[i, j, u, v] for v in self.qpus if v != u)
                            iflow = gp.quicksum(self.p[i, j, v, u] for v in self.qpus if v != u)
                            # add the cubic term constraint
                            self.model.addConstr((oflow - iflow) == 0)
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
                    self.model.addConstr(usage <= self.y[i, j] * (len(self.qpus) - 1))

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
                                
                    self.model.addConstr(self.w[u, v] <= w)
                    self.model.addConstr(w <= self.w[u, v] * len(self.qpus) * (len(self.qpus) - 1))
                    # self.model.addConstr(w <= self.w[u, v] * len(self.procs) * len(self.procs))
                    total += self.w[u, v]

        self.model.addConstr(total <= self.W)

        for u in self.qpus:
            adj = 0
            for z in self.qpus:
                if u != z:
                    if u < z:
                        adj += self.w[u, z]
                    else:
                        adj += self.w[z, u]
            self.model.addConstr(adj <= self.comms[u])
                    
    def set_obj(self):
        total = 0
        self.path_lens = {}
        for i in self.qpus:
            for j in self.qpus:
                if i < j:
                    path_len = 0
                    for u in self.qpus:
                        for v in self.qpus:
                            if u < v:
                                path_len += self.p[i, j, u, v] + self.p[i, j, v, u]
                    _demand = 0
                    for a in self.squbits:
                        for b in self.squbits:
                            if a < b and self.c[a, b] > 0:
                                _demand += self.c[a, b] * self.x[a, i] * self.x[b, j] \
                                            + self.c[a, b] * self.x[b, i] * self.x[a, j]

                    demand = self.model.addVar(vtype=gp.GRB.INTEGER)
                    self.model.addConstr(demand == _demand)
                    # self.model.addConstr(_demand >= demand)

                    total += path_len * demand

        self.model.setObjective(total, gp.GRB.MINIMIZE)



if __name__ == "__main__":
    np.random.seed(0)
    proc_num = 8
    mem = 8
    comm = 4

    qubit_num = 256
    demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = int(qubit_num * (qubit_num-1) / 6) # max
    # demand_pair = qubit_num * 2 # moderate

    qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # print(sorted(qig.demands))
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig.contract(4, inplace=True)

    # 256 homogeneous
    mems = [32, ] * 8
    comms = [8, ] * 8

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

    qig.contract_hdware_constrained(mems, inplace=True)
    # qig.contract_greedy(8, inplace=True)

    model = TACOL(qig, mems, comms, W)
    model.build()
    print(model.solve())

    # print(model.qubits_sizes)
    # print(model.c)
    # print(model.procs)


    # for a in model.qubits:
    #     for i in model.procs:
    #         print((a, i), model.x[a, i].x)
    
    # path_lengths = model.get_results()
    # print(path_lengths)