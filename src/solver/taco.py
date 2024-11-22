
import gurobipy as gp

from ..circuit.qig import QIG

from .type import ProcMemNum, ClusterMem



class TACO:
    """
    Topology-Allocation Co-Optimization
    """
    def __init__(self, qig: QIG, mems: list[ProcMemNum], W: int) -> None:
        self.qig = qig
        self.mems = mems
        self.W = W

        self.qubits = { a: node for a, node in enumerate(qig.graph.nodes) }
        self.qubits_sizes = { a: len(qig.graph.nodes[node]['qubits']) 
                                for a, node in enumerate(qig.graph.nodes) }
        self.procs = { i: mem for i, mem in enumerate(mems) }
        self.c = {}
        for a in self.qubits:
            for b in self.qubits:
                if a < b:
                    if qig.graph.has_edge(self.qubits[a], self.qubits[b]):
                        self.c[a, b] = qig.graph[self.qubits[a]][self.qubits[b]]['demand']
                    else:
                        self.c[a, b] = 0

        self.model = gp.Model()
        
    def build(self):
        pass

    def solve(self):
        pass

    def add_vars(self):
        # x[a][i] = 1 if node a is allocated to processor i
        self.x = {}
        for a in self.qubits:
            for i in self.procs:
                self.x[a, i] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'x_{a}_{i}')

        # p[i][j][u][v] = 1 if the path from node i to j goes through edge (u, v)
        self.p = {}
        for i in self.procs:
            for j in self.procs:
                if i < j:
                    for u in self.procs:
                        for v in self.procs:
                            if u != v:
                                self.p[i, j, u, v] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'p_{i}_{j}_{u}_{v}')

    def add_constrs(self):
        pass

    def add_alloc_constrs(self):
        # each node is allocated to exactly one processor
        for a in self.qubits:
            self.model.addConstr(gp.quicksum(self.x[a, i] for i in self.procs) == 1)

        # processor capacity constraint
        for i in self.procs:
            self.model.addConstr(
                # gp.quicksum(self.x[a, i] for a in self.qubits) 
                gp.quicksum(self.qubits_sizes[a] * self.x[a, i] for a in self.qubits) 
                    <= self.procs[i])
            
    def set_obj(self):
        pass