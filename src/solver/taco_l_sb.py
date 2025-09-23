
import torch
from gurobipy import GRB

import numpy as np
from src.circuit.qig import QIG, RandomQIG

from src.solver.taco_l import TACOL
from src.solver.HQCMCBD import HQCMCBD_algorithm
from src.solver.sb.converter import Model2QUBO



class TACOL_SB(TACOL):
    """"
    Solve the TACOL problem using the SB (simulated bifurcation) method.
    """

    def __init__(self, qig, mems, comms, W, timeout=600):
        super().__init__(qig, mems, comms, W, timeout)
        # self.name = "TACOL_SB"

    def solve_sb(self, config_file, penalty=1e2, max_steps=1e6, num_agents=256):
        self.model.update()
        
        # show real variable number
        var_num = 0
        for v in self.model.getVars():
            if v.vtype == GRB.BINARY and v.VarName[0] in ['x', 'y', 'p', 'w']:
                var_num += 1
        print("Variable number:", var_num)

        solver = Model2QUBO(self.model, mode = "manual", config_file=config_file)
        vec, val, obj, non_lambda = solver.run(penalty=penalty, max_steps=max_steps, num_agents=num_agents)

        updated = {}
        for j, v in enumerate(self.model.getVars()):
            if v.vtype == GRB.BINARY and v.VarName[0] in ['x', 'p', 'y', 'w'] and v.VarName in solver.Bin_varname:
                self.model.getVarByName(v.VarName).lb = int(vec[solver.Bin_varname.index(v.VarName)])
                self.model.getVarByName(v.VarName).ub = int(vec[solver.Bin_varname.index(v.VarName)])
                updated[v.VarName] = int(vec[solver.Bin_varname.index(v.VarName)])

        print("Updated variables:", updated)
        self.model.update()
        self.model.optimize()
        print("Gurobi model updated with best sample values.")
        print("Gurobi model objective value:", self.model.getObjective().getValue())

    def solve_sa(self, config_file=None):
        from src.solver.sb.cqm import CQMConverter
        # use simulated bifurcation to solve the model
        # deep copy the model
        from copy import deepcopy
        # model = deepcopy(self.model)
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
            if varname in best_sample and varname[0] in ['x', 'p', 'y', 'w']:
                self.model.getVarByName(varname).lb = int(best_sample[varname])
                self.model.getVarByName(varname).ub = int(best_sample[varname])
            else:
                print(f"Variable {varname} not found in best sample.")

        self.model.update()
        self.model.optimize()
        print("Gurobi model updated with best sample values.")
        print("Gurobi model objective value:", self.model.getObjective().getValue())

    def solve_sa1(self, config_file=None):
        from src.solver.sb.cqm import CQMConverter
        # use simulated bifurcation to solve the model
        solver = CQMConverter(self.model, config_file=config_file, mode="manual")
        print("Number of binary variables in model:", self.model.NumBinVars)
        print("Number of binary variables:", solver.num_binvars)
        cqm = solver.run()
        
        import dimod
        # solve the CQM using simulated bifurcation
        bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=100000)
        # print(bqm.variables)
        # print("inverter:")
        # print(invert.to_dict())
        
        # convert to QUBO
        Q, offset = bqm.to_qubo()
        from src.utils.qubo import QDict2Matrix
        Q_tensor = QDict2Matrix(Q)
        # print("Q matrix:\n", Q_tensor)
        import simulated_bifurcation as sb
        vec, val = sb.minimize(
            matrix=Q_tensor, constant=offset, best_only=True, domain='binary',
            max_steps=1e5, agents=256, device="cuda"
            )
        print("Best value:", val)
        # print("Best vector:", vec)

        updated = {}
        for j, v in enumerate(self.model.getVars()):
            if v.vtype == GRB.BINARY and v.VarName in solver.Bin_varname:
                self.model.getVarByName(v.VarName).lb = int(vec[solver.Bin_varname.index(v.VarName)])
                self.model.getVarByName(v.VarName).ub = int(vec[solver.Bin_varname.index(v.VarName)])
                updated[v.VarName] = int(vec[solver.Bin_varname.index(v.VarName)])
        print("Updated variables:", updated)
        self.model.update()
        self.model.optimize()
        print("Gurobi model updated with best sample values.")
        print("Gurobi model objective value:", self.model.getObjective().getValue())


if __name__ == "__main__":

    PROJ_DIR = '/home/ljy/projects/topodqc/'
    config_file = 'src/solver/sb/config.json'
    config_file = PROJ_DIR + config_file

    # set torch default tensor type to double
    torch.set_default_tensor_type(torch.FloatTensor)
    # use GPU as default device
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(0)
    proc_num = 4
    mem = 8
    comm = 4

    qubit_num = 32
    demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = qubit_num * 2 # moderate

    qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    # qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig.contract(4, inplace=True)

    mems = [mem] * proc_num
    comms = [comm] * proc_num
    W = int(proc_num * (proc_num-1) / 2)
    print("W:", W)
    # W = (proc_num - 1)
    # W = proc_num * 2

    model = TACOL_SB(qig, mems, comms, W)
    model.build()
    model.model.update()

    # model.solve()

    model.solve_sb(config_file, penalty=1e4, max_steps=5e4, num_agents=256)

    # model.solve_sa(config_file)
    # model.solve_sa1(config_file)

