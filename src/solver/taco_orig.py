


import gurobipy as gp
import numpy as np

from .taco import TACO
from ..circuit.qig import QIG, RandomQIG
from .type import ProcMemNum


class TACOORIG(TACO):
    def __init__(self, 
            qig: QIG, 
            mems: list[ProcMemNum], 
            comms: list[ProcMemNum], 
            W: int,
            timeout: int = 600
            ) -> None:
        super().__init__(qig, mems, comms, W, timeout)
        # set NonConvex to 2 for non-convex quadratic constraints
        # self.model.setParam('NonConvex', 2)

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
                                y_expr += self.x[a, i] * self.x[b, j] + self.x[a, j] * self.x[b, i]
                    
                    # for start node i
                    self.y[i, j] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f'y_{i}_{j}')
                    self.model.addConstr(self.y[i, j] == y_expr)
                    
                    oflow = gp.quicksum(self.p[i, j, i, u] for u in self.qpus if u != i)
                    iflow = gp.quicksum(self.p[i, j, u, i] for u in self.qpus if u != i)
                    self.model.addConstr(self.y[i, j] * (oflow - iflow) == self.y[i, j])
                    # for end node j
                    oflow = gp.quicksum(self.p[i, j, j, u] for u in self.qpus if u != j)
                    iflow = gp.quicksum(self.p[i, j, u, j] for u in self.qpus if u != j)
                    self.model.addConstr(self.y[i, j] * (oflow - iflow) == -self.y[i, j])
                    # for intermediate nodes
                    for u in self.qpus:
                        if u != i and u != j:
                            oflow = gp.quicksum(self.p[i, j, u, v] for v in self.qpus if v != u)
                            iflow = gp.quicksum(self.p[i, j, v, u] for v in self.qpus if v != u)
                            self.model.addConstr(oflow - iflow == 0)
                    
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
                    self.model.addConstr(usage <= self.y[i, j] * (len(self.qpus) - 1))

    def add_topology_constrs(self):
        # total edge number in the topology
        total = 0
        for u in self.qpus:
            for v in self.qpus:
                if u < v:
                    prev = 1
                    for i in self.qpus:
                        for j in self.qpus:
                            if i < j:
                                _w = self.model.addVar(vtype=gp.GRB.BINARY)
                                self.model.addConstr(_w == (1 - self.p[i, j, u, v]) * (1 - self.p[i, j, v, u]))
                                
                                temp = self.model.addVar(vtype=gp.GRB.BINARY)
                                self.model.addConstr(temp == prev * _w)
                                
                                prev = temp

                    total += 1 - prev

        self.model.addConstr(total <= self.W)

        # max adjacent edge for each processor
        for u in self.qpus:
            adj = 0
            for z in self.qpus:
                if u != z:
                    unused = 1
                    for i in self.qpus:
                        for j in self.qpus:
                            if i < j:
                                _e = self.model.addVar(vtype=gp.GRB.BINARY)
                                self.model.addConstr(_e == (1 - self.p[i, j, u, z]) * (1 - self.p[i, j, z, u]))
                                
                                temp = self.model.addVar(vtype=gp.GRB.BINARY)
                                self.model.addConstr(temp == unused * _e)
                                
                                unused = temp

                    adj += 1 - unused
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

    def export(self, path: str):
        self.model.write(path)




if __name__ == "__main__":
    np.random.seed(0)
    proc_num = 2
    mem = 4
    comm = 4

    qubit_num = 8
    demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = qubit_num * 2 # moderate
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # qig.contract(4, inplace=True)

    mems = [mem] * proc_num
    comms = [comm] * proc_num
    W = proc_num * (proc_num-1) / 2
    # W = (proc_num - 1) 

    model = TACOORIG(qig, mems, comms, W)
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
    # {(0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0, (1, 2): 1.0, (1, 3): 1.0, (2, 3): 0.0}
    # {(0, 1): 0.0, (0, 2): 1.0, (0, 3): 1.0, (1, 2): 1.0, (1, 3): 1.0, (2, 3): 2.0}