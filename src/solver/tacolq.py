


import gurobipy as gp
import numpy as np

from .taco import TACO
from ..circuit.qig import QIG, RandomQIG
from .type import ProcMemNum



class LinearFormu(TACO):
    def __init__(self, qig: QIG, mems: list[ProcMemNum], W: int) -> None:
        super().__init__(qig, mems, W)

        # set NonConvex to 1
        # self.model.setParam('NonConvex', 2)

    def build(self):
        self.add_vars()
        self.add_alloc_constrs()
        self.add_path_constrs()
        self.add_connectivity_constrs()
        self.set_obj()


    def solve(self):
        self.model.update()
        self.model.optimize()
        
        if hasattr(self.model, 'objVal'):
            return self.model.objVal
        else:
            return None

    def add_path_constrs(self):
        # each node is allocated to exactly one processor
        self.y = {}
        for i in self.procs:
            for j in self.procs:
                if i < j:
                    y_expr = gp.QuadExpr(0)
                    for a in self.qubits:
                        for b in self.qubits:
                            if a < b and self.c[a, b] > 0:
                                y_expr += self.x[a, i] * self.x[b, j] + self.x[b, i] * self.x[a, j]

                    self.y[i, j] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}_{j}')
                    self.model.addConstr(self.y[i, j] <= y_expr)
                    self.model.addConstr(y_expr <= self.y[i, j] * (len(self.qubits) // 2))
                    # self.model.addConstr(y <= self.y[i, j] * len(self.qubits))
                    
                    # for start node i
                    oflow = gp.quicksum(self.p[i, j, i, u] for u in self.procs if u != i)
                    iflow = gp.quicksum(self.p[i, j, u, i] for u in self.procs if u != i)
                    self.model.addConstr((oflow - iflow) == self.y[i, j])
                    # self.model.addConstr(oflow - iflow == 1)
                    # for end node j
                    oflow = gp.quicksum(self.p[i, j, j, u] for u in self.procs if u != j)
                    iflow = gp.quicksum(self.p[i, j, u, j] for u in self.procs if u != j)
                    self.model.addConstr((oflow - iflow) == -self.y[i, j])
                    # self.model.addConstr(oflow - iflow == -1)
                    # for intermediate nodes
                    for u in self.procs:
                        if u != i and u != j:
                            oflow = gp.quicksum(self.p[i, j, u, v] for v in self.procs if v != u)
                            iflow = gp.quicksum(self.p[i, j, v, u] for v in self.procs if v != u)
                            # add the cubic term constraint
                            self.model.addConstr((oflow - iflow) == 0)
                            # self.model.addConstr(oflow - iflow == 0)
                    
        # cancel no demand path
        for i in self.procs:
            for j in self.procs:
                if i < j:
                    usage = 0
                    for u in self.procs:
                        for v in self.procs:
                            if u < v:
                                usage += self.p[i, j, u, v] + self.p[i, j, v, u]
                                # self.model.addConstr(self.p[i, j, u, v] <= self.y[i, j])
                                # self.model.addConstr(self.p[i, j, v, u] <= self.y[i, j])
                                # self.model.addConstr(self.p[i, j, u, v] + self.p[i, j, v, u] <= self.y[i, j])
                    self.model.addConstr(usage <= self.y[i, j] * (len(self.procs) - 1))


    def add_connectivity_constrs(self):
        self.w = {}
        total = 0
        for u in self.procs:
            for v in self.procs:
                if u < v:
                    self.w[u, v] = self.model.addVar(vtype=gp.GRB.BINARY)
                    w = 0
                    for i in self.procs:
                        for j in self.procs:
                            if i < j:
                                w += self.p[i, j, u, v] + self.p[i, j, v, u]
                                
                    self.model.addConstr(self.w[u, v] <= w)
                    self.model.addConstr(w <= self.w[u, v] * len(self.procs) * (len(self.procs) - 1))
                    # self.model.addConstr(w <= self.w[u, v] * len(self.procs) * len(self.procs))
                    total += self.w[u, v]

        self.model.addConstr(total <= self.W)
             

    def set_obj(self):
        total = 0
        self.path_lens = {}
        for i in self.procs:
            for j in self.procs:
                if i < j:
                    path_len = 0
                    for u in self.procs:
                        for v in self.procs:
                            if u < v:
                                path_len += self.p[i, j, u, v] + self.p[i, j, v, u]
                    _demand = 0
                    for a in self.qubits:
                        for b in self.qubits:
                            if a < b and self.c[a, b] > 0:
                                _demand += self.c[a, b] * self.x[a, i] * self.x[b, j] \
                                            + self.c[a, b] * self.x[b, i] * self.x[a, j]

                    demand = self.model.addVar(vtype=gp.GRB.INTEGER)
                    self.model.addConstr(demand == _demand)
                    # self.model.addConstr(_demand >= demand)

                    total += path_len * demand

        self.model.setObjective(total, gp.GRB.MINIMIZE)


    def get_results(self):
        path_lengths = {}
        for i in self.procs:
            for j in self.procs:
                if i < j:
                    path_len = 0
                    for u in self.procs:
                        for v in self.procs:
                            if u < v:
                                path_len += self.p[i, j, u, v].x + self.p[i, j, v, u].x
                    path_lengths[i, j] = path_len

        return path_lengths


if __name__ == "__main__":
    np.random.seed(0)
    qubit_num = 64
    demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    demand_pair = qubit_num * 2 # moderate
    proc_num = 8
    mem = 8
    qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    qig.contract(4, inplace=True)

    mems = [mem] * proc_num
    W = proc_num * (proc_num-1) / 2
    # W = (proc_num - 1) 

    model = LinearFormu(qig, mems, W)
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