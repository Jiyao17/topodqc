
import torch
import gurobipy as gp
from gurobipy import GRB
import dimod
import simulated_bifurcation as sb

import numpy as np
from src.circuit.qig import QIG, RandomQIG

from src.solver.taco_l import TACOL
from src.solver.HQCMCBD import HQCMCBD_algorithm
from src.solver.sb.converter import Model2QUBO
from src.solver.sb.cqm import CQMConverter
from src.utils.qubo import QDict2Matrix



class SimpleQUBO():

    def __init__(self):
        self.model = gp.Model("SimpleQUBO")

    def build(self):
        # build a simple QUBO model:
        # min x1 + x2 + x3
        # s.t. x1, x2, x3 binary
        # x1 + x2 + x3 <= 1
        x1 = self.model.addVar(vtype=GRB.BINARY, name="x1")
        x2 = self.model.addVar(vtype=GRB.BINARY, name="x2")
        x3 = self.model.addVar(vtype=GRB.BINARY, name="x3")

        self.model.setObjective(x1 + x2 + x3, GRB.MINIMIZE)
        self.model.addConstr(x1 + x2 >= 1, "c1")
        self.model.addConstr(x2 + x3 >= 1, "c2")
        self.model.update()


    def solve(self, config_file=None):
        # use simulated bifurcation to solve the model
        solver = Model2QUBO(self.model, config_file=config_file, mode="manual")
        vec, val, obj, non_lambda = solver.run(penalty=1000, max_steps=5e4)

        print("Solution vector:", vec)
        print("Solution value:", val)
        print("Objective value:", obj)
        print("Non-lambda objective value:", non_lambda)

    def solve_cqm(self, config_file=None):
        # use simulated bifurcation to solve the model
        solver = CQMConverter(self.model, config_file=config_file, mode="manual")
        cqm = solver.run()
        
        # solve the CQM using simulated bifurcation
        bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)
        Q, offset = bqm.to_qubo()
        Q_tensor = QDict2Matrix(Q)
        print("Q matrix:\n", Q_tensor)
        vec, val = sb.minimize(matrix=Q_tensor, constant=offset, best_only=True, domain='binary')
        print("Best value:", val)
        print("Best vector:", vec)

    def solve_sa(self, config_file=None):
        from src.solver.sb.cqm import CQMConverter
        # use simulated bifurcation to solve the model
        solver = CQMConverter(self.model, config_file=config_file, mode="manual")
        print("Number of binary variables in model:", self.model.NumBinVars)
        print("Number of binary variables:", solver.num_binvars)
        cqm = solver.run()
        
        import dimod
        # solve the CQM using simulated bifurcation
        bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=100)
        # print(bqm.variables)
        # print("inverter:")
        # print(invert.to_dict())
        
        # solve the BQM using simulated annealing
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=10)
        # Print the best sample
        best_sample = sampleset.first.sample
        print("Best sample:", best_sample)
        print("Objective value:", sampleset.first.energy)

        var_num = len(self.model.getVars())
        print("Variable number in Gurobi model:", var_num)
        print("Best sample size:", len(best_sample))
        # convert back to gurobi model objective
        for i, var in enumerate(self.model.getVars()):
            varname = var.VarName
            if varname in best_sample:
                self.model.getVarByName(varname).lb = int(best_sample[varname])
                self.model.getVarByName(varname).ub = int(best_sample[varname])
            else:
                print(f"Variable {varname} not found in best sample.")

        self.model.update()
        self.model.optimize()
        print("Gurobi model updated with best sample values.")
        print("Gurobi model objective value:", self.model.getObjective().getValue())


def test():
    import dimod

    # formulate a binary cubic problem with quadratic constraints
    # min   -xyz + xy + xz - yz - x
    # s.t.  x + y >= z
    #       x*y <= z

    # BPM -> CQM -> BQM -> QUBO -> ISING
    #                        Q  -> SB

    # 1. Create BinaryPolynomial objective and constraints
    # Cubic Objective: -xyz + xy + xz - yz - x
    x, y, z = dimod.Binaries(('x', 'y', 'z'))
    poly_obj = {('x',): 1, ('y',): 1, ('z',): 1}
    poly_obj = dimod.BinaryPolynomial(poly_obj, vartype=dimod.BINARY)
    # print(type(poly_obj.variables.pop()))

    # Quadratic Constraints: x*y <= z
    # x*y <= z ===> x*y - z <= 0 
    # w = x*y in a linear way so x*y - z <= 0 =====> w - z <= 0
    # now find w:
    # w - x <= 0
    # w - y <= 0
    # w - x - y + 1 >= 0


    # 2. Convert to quadratic (CQM)
    # a penalty strength here
    quad_obj = dimod.make_quadratic(poly_obj, strength=10, vartype=dimod.BINARY)


    cqm = dimod.CQM()
    # Add the objective function to the CQM
    cqm.set_objective(quad_obj)
    # add constraint x + y < z
    cqm.add_constraint(x + y + z >= 1)
    # cqm.add_constraint(x * y - z<= 0)
    # cqm.add_constraint(w - z <= 0)
    # cqm.add_constraint(w - x <= 0)
    # cqm.add_constraint(w - y <= 0)
    # cqm.add_constraint(w - x - y + 1 >= 0)

    # may solve CQM with quadratic constraints with LeapHybridCQMSampler
    # BQM does not allow the original CQM have quadratic constraints

    # CQM -> BQM
    # a penalty strength here
    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)

    # BQM -> QUBO
    Q, offset = bqm.to_qubo()

    # Solve the QUBO using Simulated Bifurcation
    import simulated_bifurcation as sb
    import torch
    import math

    def QDict2Matrix(Q_dict: dict) -> torch.Tensor:
        """
        Convert a dictionary of quadratic terms to a matrix.
        Each key in the dictionary is a tuple (i, j) representing the variables,
        and the value is the coefficient for that term.
        """
        
        keys = list(Q_dict.keys())
        variables = set()
        for i, j in keys:
            variables.add(i)
            variables.add(j)
        idx_map = {var: idx for idx, var in enumerate(sorted(variables))}
        size = len(idx_map)
        matrix = torch.zeros((size, size), dtype=torch.float32)
        for (i, j), value in Q_dict.items():
            matrix[idx_map[i], idx_map[j]] = value
            
        return matrix

    Q = QDict2Matrix(Q)
    print("Q matrix:\n", Q)
    # print("Q size:", Q.size())
    vec, val = sb.minimize(matrix=Q, constant=offset, best_only=True, domain='binary',)
    # show the result
    print("Best value:", val)
    print("Best vector:", vec)


    # # solve the BQM using simulated annealing
    from dimod import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=10)

    # # Print the results
    # print(sampleset)
    # # Print the best sample
    best_sample = sampleset.first.sample
    print("Best sample:", best_sample)

    print("Objective value:", sampleset.first.energy)
    # print("Objective value:", bqm.energy(best_sample))

def test_pyQUBO():
    from pyqubo import Binary, Constraint, solve_qubo

    # Define binary variables
    x = Binary('x')
    y = Binary('y')
    z = Binary('z')

    # Define the objective function
    # objective = x*y*z + x*y - x*z + y*z + x
    objective = x + y + z

    # Define the constraints
    constraint1 = Constraint(x + y + z - 1, label='c1', condition=lambda x: x >= 0)
    # constraint1 = Constraint(-x - y - z + 1, label='c2')

    # Create the model
    model = objective + 10* constraint1
    # model = objective

    # Compile the model to QUBO
    bqm = model.compile().to_bqm()

    # Solve the QUBO using simulated annealing
    # # solve the BQM using simulated annealing
    from dimod import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=10)

    # # Print the results
    # print(sampleset)
    # # Print the best sample
    best_sample = sampleset.first.sample
    print("Best sample:", best_sample)

    print("Objective value:", sampleset.first.energy)
    # print("Objective value:", bqm.energy(best_sample))

if __name__ == "__main__":

    PROJ_DIR = '/home/ljy/projects/topodqc/'
    config_file = 'src/solver/sb/config.json'
    config_file = PROJ_DIR + config_file

    model = SimpleQUBO()
    model.build()
    model.solve(config_file=config_file)
    # model.solve_cqm(config_file=config_file)
    # model.solve_sa(config_file=config_file)

    # test()
    # test_pyQUBO()

