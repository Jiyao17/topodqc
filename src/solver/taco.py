
import gurobipy as gp

from ..circuit.qig import QIG

from .type import ProcMemNum, ClusterMem, ProcCommNum

class TACO:
    pass





class TACO:
    """
    Topology-Allocation Co-Optimization
    """

    OBJ_VALS = []

    def callback(model: gp.Model, where=gp.GRB.Callback.MIPSOL):
        """
        callback function to record the objective values and time
        """
        if where == gp.GRB.Callback.MIPSOL:
            time = model.cbGet(gp.GRB.Callback.RUNTIME)
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)

            TACO.OBJ_VALS.append((time, obj))

    def __init__(self, 
            qig: QIG, 
            mems: list[ProcMemNum], 
            comms: list[ProcCommNum],
            W: int,
            timeout: int = 300
            ) -> None:
        TACO.OBJ_VALS = []

        self.qig = qig
        # memory size of each processor
        self.mems = mems
        # communication qubit number of each processor
        self.comms = comms
        self.W = W
        self.timeout = timeout

        self.squbits = { a: node for a, node in enumerate(qig.graph.nodes) }
        self.squbits_rev = { node: a for a, node in enumerate(qig.graph.nodes) }
        self.squbits_sizes = { a: len(qig.graph.nodes[node]['qubits']) 
                                for a, node in enumerate(qig.graph.nodes) }
        self.qpus = { i: mem for i, mem in enumerate(mems) }
        # self.qpus_rev = { mem: i for i, mem in enumerate(mems) }

        self.edge_weights = {}

        self.c = {}
        for a in self.squbits:
            for b in self.squbits:
                if a < b:
                    if qig.graph.has_edge(self.squbits[a], self.squbits[b]):
                        self.c[a, b] = qig.graph[self.squbits[a]][self.squbits[b]]['demand']
                    else:
                        self.c[a, b] = 0

        self.model = gp.Model()

        self.model.setParam('TimeLimit', self.timeout)

    def build(self):
        pass

    def solve(self, callback=callback):
        self.model.update()
        self.model.optimize(callback=callback)
        
        if hasattr(self.model, 'objVal'):
            return self.model.objVal
        else:
            return None
        
    def add_vars(self):
        # x[a][i] = 1 if node a is allocated to processor i
        self.x = {}
        for a in self.squbits:
            for i in self.qpus:
                self.x[a, i] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'x_{a}_{i}')

        # p[i][j][u][v] = 1 if the path from node i to j goes through edge (u, v)
        self.p = {}
        for i in self.qpus:
            for j in self.qpus:
                if i < j:
                    for u in self.qpus:
                        for v in self.qpus:
                            if u != v:
                                self.p[i, j, u, v] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'p_{i}_{j}_{u}_{v}')

    def add_constrs(self):
        pass

    def add_alloc_constrs(self):
        # each node is allocated to exactly one processor
        for a in self.squbits:
            self.model.addConstr(gp.quicksum(self.x[a, i] for i in self.qpus) == 1, name=f'alloc_constr_{a}')

        # processor capacity constraint
        for i in self.qpus:
            self.model.addConstr(
                # gp.quicksum(self.x[a, i] for a in self.qubits) 
                gp.quicksum(self.squbits_sizes[a] * self.x[a, i] for a in self.squbits) 
                    <= self.qpus[i],
                name=f'proc_mem_constr_{i}')

    def set_obj(self):
        pass

    def get_topology(self):
        if len(TACO.OBJ_VALS) == 0:
            return None
        
        edges = []
        for u in self.qpus:
            for v in self.qpus:
                if u < v:
                    for i in self.qpus:
                        for j in self.qpus:
                            if i < j:
                                if self.p[i, j, u, v].x > 0.5 or self.p[i, j, v, u].x > 0.5:
                                    edges.append((u, v))

        return edges
    
    def get_objs(self):
        if len(TACO.OBJ_VALS) == 0:
            return None
        else:
            return TACO.OBJ_VALS

