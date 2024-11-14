
import gurobipy as gp

from .solver import TopoAllocCoOp
from ..circuit.qig import QIG, RandomQIG
from .type import ProcMemNum



class OrigFormu(TopoAllocCoOp):
    def __init__(self, qig: QIG, mems: list[ProcMemNum], W: int) -> None:
        super().__init__(qig, mems, W)

        # set NonConvex to 1
        self.model.setParam('NonConvex', 1)

    def build(self):
        self.add_vars()
        self.add_constrs()
        self.set_obj()

        self.model.update()

    def solve(self):
        self.model.optimize()
        
        if hasattr(self.model, 'objVal'):
            return self.model.objVal
        else:
            return None

    def add_constrs(self):
        self.add_alloc_constrs()
        self.add_path_constrs()
        self.add_connectivity_constrs()

    def add_path_constrs(self):
        # each node is allocated to exactly one processor
        for i in self.procs:
            for j in self.procs:
                if i < j:
                    _exist = 0
                    for a in self.qubits:
                        for b in self.qubits:
                            if a < b and self.c[a, b] > 0:
                                _exist += self.x[a, i] * self.x[b, j] + self.x[b, i] * self.x[a, j]
                    exist = self.model.addVar(vtype=gp.GRB.BINARY, name=f'I_{i}_{j}')
                    self.model.addConstr(exist == _exist)
                    # self.model.addConstr(exist >= _exist)

                    # for start node i
                    oflow = gp.quicksum(self.p[i, j, i, u] for u in self.procs if u != i)
                    iflow = gp.quicksum(self.p[i, j, u, i] for u in self.procs if u != i)
                    self.model.addConstr(exist * (oflow - iflow) == exist)
                    # self.model.addConstr(oflow - iflow == 1)
                    # for end node j
                    oflow = gp.quicksum(self.p[i, j, j, u] for u in self.procs if u != j)
                    iflow = gp.quicksum(self.p[i, j, u, j] for u in self.procs if u != j)
                    self.model.addConstr(exist * (oflow - iflow) == -exist)
                    # self.model.addConstr(oflow - iflow == -1)
                    # for intermediate nodes
                    for u in self.procs:
                        if u != i and u != j:
                            oflow = gp.quicksum(self.p[i, j, u, v] for v in self.procs 
                                                if v != u)
                            iflow = gp.quicksum(self.p[i, j, v, u] for v in self.procs 
                                                if v != u)
                            # add the cubic term constraint
                            self.model.addConstr(exist * (oflow - iflow) == 0)
                            # self.model.addConstr(oflow - iflow == 0)

    def add_connectivity_constrs(self):
        total = 0
        for u in self.procs:
            for v in self.procs:
                if u != v:
                    not_used = 1
                    for i in self.procs:
                        for j in self.procs:
                            if i < j:
                                term = self.model.addVar(vtype=gp.GRB.BINARY)
                                self.model.addConstr(term == (1 - self.p[i, j, u, v]) * (1 - self.p[i, j, v, u]))
                                # self.model.addConstr(term >= (1 - self.p[i, j, u, v]) + (1 - self.p[i, j, v, u]))
                                
                                new_not_used = self.model.addVar(vtype=gp.GRB.BINARY)
                                self.model.addConstr(new_not_used == not_used * term)
                                # self.model.addConstr(new_not_used >= not_used * term)

                                not_used = new_not_used
                    used = 1 - not_used
                    total += used

        self.model.addConstr(total <= self.W)

    def set_obj(self):
        total = 0
        for i in self.procs:
            for j in self.procs:
                if i != j:
                    path_len = 0
                    for u in self.procs:
                        for v in self.procs:
                            if u != v:
                                path_len += self.p[i, j, u, v] + self.p[i, j, v, u]
                    demand = 0
                    for a in self.qubits:
                        for b in self.qubits:
                            if a < b:
                                demand += self.c[a, b] * self.x[a, i] * self.x[b, j]

                    _demand = self.model.addVar(vtype=gp.GRB.BINARY)
                    self.model.addConstr(_demand <= demand)
                    self.model.addConstr(_demand >= demand)

                    total += path_len * _demand

        self.model.setObjective(total, gp.GRB.MINIMIZE)


if __name__ == "__main__":
    qubit_num = 8
    proc_num = 2
    mem = 4
    qig = RandomQIG(qubit_num, qubit_num*2, (1, 10))
    mems = [mem] * proc_num
    W = proc_num * proc_num

    model = OrigFormu(qig, mems, W)
    model.build()
    print(model.solve())

    # print(model.c)