


# Convert Gurobi models to kaiwu QuboModel 
# so it can be solved by kaiwu simulated annealing solver
# kaiwu SA solver offers immediate original objective value

import gurobipy as gp
import kaiwu as kw
import numpy as np
import time

from kaiwu.qubo import QuboModel, QuboExpression
from kaiwu.classical import SimulatedAnnealingOptimizer, TabuSearchOptimizer

from src.solver.taco_l import TACOL
from src.circuit.qig import QIG



class Gurobi2Qubo(object):
    """
    Convert a Gurobi model to a kaiwu QuboModel.
    """
    def __init__(self, gurobi_model: gp.Model, slack_bound: int=127):
        self.gurobi_model = gurobi_model
        self.slack_bound = slack_bound
        self.qubo_model = QuboModel()

        self.qvars = {}

    def _convert_vars(self):
        # Extract variables and their coefficients
        vars = self.gurobi_model.getVars()
        # only binary and integer variables are supported
        assert all(v.VType in [gp.GRB.BINARY, gp.GRB.INTEGER] for v in vars), "Only binary and integer variables are supported"
        # assert all(v.VType in [gp.GRB.BINARY] for v in vars), "Only binary variables are supported"

        # add a corresponding kaiwu variable for each gurobi variable
        self.qvars = {}
        for v in vars:
            if v.VType == gp.GRB.BINARY:
                self.qvars[v.VarName] = kw.qubo.Binary(v.VarName)
            elif v.VType == gp.GRB.INTEGER:
                self.qvars[v.VarName] = kw.qubo.Integer(v.VarName, int(v.LB), int(v.UB))

    def _convert_objective(self) -> QuboExpression:
        """
        Convert a Gurobi objective function to a kaiwu QuboExpression.
        """
        obj = self.gurobi_model.getObjective()

        qexpr = 0
        if isinstance(obj, gp.QuadExpr):
            for i in range(obj.size()):
                var1 = obj.getVar1(i)
                var2 = obj.getVar2(i)
                coeff = obj.getCoeff(i)
                qexpr += coeff * self.qvars[var1.VarName] * self.qvars[var2.VarName]

            obj = obj.getLinExpr()
        
        for i in range(obj.size()):
            var = obj.getVar(i)
            coeff = obj.getCoeff(i)
            qexpr += coeff * self.qvars[var.VarName]

        return qexpr

    def _convert_linear_constr(self, constr: gp.Constr) -> QuboExpression:
        """
        Convert a Gurobi linear constraint to a kaiwu QuboExpression.
        """
        expr = self.gurobi_model.getRow(constr)
        sense = constr.Sense
        rhs = constr.RHS
        name = constr.ConstrName

        # Build the left-hand side expression
        lhs_expr = 0
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            lhs_expr += coeff * self.qvars[var.VarName]
            

        # Convert to QuboExpression based on the sense of the constraint
        if sense == gp.GRB.LESS_EQUAL:
            slack = kw.qubo.Integer(f"slack_{name}", 0, self.slack_bound)
            qexpr = lhs_expr - rhs + slack
        elif sense == gp.GRB.GREATER_EQUAL:
            slack = kw.qubo.Integer(f"slack_{name}", 0, self.slack_bound)
            qexpr = rhs - lhs_expr + slack
        elif sense == gp.GRB.EQUAL:
            qexpr = (lhs_expr - rhs) ** 2
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")
        
        return qexpr
        
    def _convert_quadratic_constr(self, constr: gp.QConstr) -> QuboExpression:
        """
        Convert a Gurobi quadratic constraint to a kaiwu QuboExpression.
        """
        expr = self.gurobi_model.getQCRow(constr)
        sense = constr.QCSense
        rhs = constr.QCRHS
        name = constr.QCName

        # Build the left-hand side expression
        # quadratic part first
        lhs_qexpr = 0
        for i in range(expr.size()):
            var1 = expr.getVar1(i)
            var2 = expr.getVar2(i)
            coeff = expr.getCoeff(i)
            lhs_qexpr += coeff * self.qvars[var1.VarName] * self.qvars[var2.VarName]

        # linear part
        lhs_lexpr = 0
        linexpr = expr.getLinExpr()
        for i in range(linexpr.size()):
            var = linexpr.getVar(i)
            coeff = linexpr.getCoeff(i)
            lhs_lexpr += coeff * self.qvars[var.VarName]

        lhs_expr = lhs_qexpr + lhs_lexpr

        # Convert to QuboExpression based on the sense of the constraint
        if sense == gp.GRB.LESS_EQUAL:
            slack = kw.qubo.Integer(f"slack_{name}", 0, self.slack_bound)
            qexpr = lhs_expr - rhs + slack
            # qexpr = (lhs_expr - rhs) ** 2
        elif sense == gp.GRB.GREATER_EQUAL:
            slack = kw.qubo.Integer(f"slack_{name}", 0, self.slack_bound)
            qexpr = rhs - lhs_expr + slack
        elif sense == gp.GRB.EQUAL:
            qexpr = (lhs_expr - rhs) ** 2
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")
        
        return qexpr

    def convert(self, penalty=10) -> QuboModel:
        """
        Convert the Gurobi model to a kaiwu QuboModel.
        """
        self._convert_vars()
        obj_expr = self._convert_objective()
        self.qubo_model.set_objective(obj_expr)

        constrs = self.gurobi_model.getConstrs()
        qconstrs = self.gurobi_model.getQConstrs()
        for constr in constrs:
            qubo_constr = self._convert_linear_constr(constr) * penalty
            self.qubo_model.add_constraint(qubo_constr, constr.ConstrName)
        for qconstr in qconstrs:
            qubo_constr = self._convert_quadratic_constr(qconstr) * penalty
            self.qubo_model.add_constraint(qubo_constr, qconstr.QCName)
        # self.qubo_model.make()
        # print(qubo_model)

        # self.qubo_model.get_matrix()  # to make sure the Q matrix is generated
        # self.qubo_model.get_offset()  # to make sure the offset is generated

        return self.qubo_model

    def revert(self, qubo_sol: dict) -> dict:
        """
        Revert a kaiwu QuboModel solution to a Gurobi solution.
        """
        assert qubo_sol is not None, "No solution to revert"
        # models containing Integers are not supported yet
        assert all(v.VType == gp.GRB.BINARY for v in self.gurobi_model.getVars()), "Only binary variables are supported"

        gurobi_sol = {}
        for var_name, qvar in self.qvars.items():
            if self.gurobi_model.getVarByName(var_name).VType == gp.GRB.BINARY:
                gurobi_sol[var_name] = qubo_sol[var_name]
            elif self.gurobi_model.getVarByName(var_name).VType == gp.GRB.INTEGER:
                gurobi_sol[var_name] = int(qubo_sol[var_name])

        return gurobi_sol


class SASolver(object):
    """
    Solve a QuboModel using kaiwu simulated annealing solver.
    """
    def __init__(self, process_num: int=4):
        self.process_num = process_num
        self.sa = None
    
    def set_new_sa(self, seed):
        self.sa = SimulatedAnnealingOptimizer(
            initial_temperature=1000,
            alpha=0.99,
            cutoff_temperature=1e-1,
            iterations_per_t=50,
            size_limit=100000,
            process_num=self.process_num,
            rand_seed=seed
            )


    def solve(self, 
            gurobi_model: gp.Model, 
            slack_bound: int=127, 
            timeout: int=300
            ):
        converter = Gurobi2Qubo(gurobi_model, slack_bound)
        qubo_model: QuboModel = converter.convert(penalty=100)
        Q = qubo_model.get_matrix()
        J, b = kw.core.qubo_matrix_to_ising_matrix(Q)
        print("qubo variables:", len(qubo_model.get_variables()))
        print("qubo constraints:", len(qubo_model.get_constraints()))

        best_sol = None
        best_val = float("inf")
        best_qubo_val = float("inf")
        best_violation = float("inf")
        start_time = time.time()
        solutions = []
        solving_num = 0
        while time.time() - start_time < timeout:
            seed = int(time.time() * 1000) % 100000
            self.set_new_sa(seed)
            # results = self.sa.solve(J, )
            print(f"Solving round {solving_num+1} with seed {seed}...")
            results = self.sa.solve(J, best_sol)
            print(f"Analyzing {len(results)} solutions.")
            for res in results:
                # res = kw.sampler.spin_to_binary(res)
                sol_dict = qubo_model.get_sol_dict(res)
                val = qubo_model.get_value(sol_dict)
                violation, vio_dict = qubo_model.verify_constraint(sol_dict)
                if violation < best_violation or (violation == best_violation and val < best_val):
                    best_violation = violation
                    best_val = val
                    best_sol = res
                    
                    elapsed = time.time() - start_time
                    solutions.append((elapsed, best_val, best_violation))
                    print(f"New best solution found at {elapsed:.2f}s: obj={best_val}, violation={best_violation}")
                    # print(sol_dict)
            
            solving_num += 1
            elapsed = time.time() - start_time
            print(f"Solving round {solving_num} finished at {elapsed:.2f}s, {len(results)} solutions explored.")
            
            if best_violation == 0:
                print("Feasible solution found.")
                break

        return solutions


class TACOSA(object):
    """
    Wrapper for SASolver to be used in the TACO framework.
    """
    def __init__(self, qig: QIG, mems: list, comms: list, W: int, timeout: int=600):
        self.qig = qig
        self.mems = mems
        self.comms = comms
        self.W = W
        self.timeout = timeout
        self.tacol = None
        self.qubo_model = None

        self.solutions = None

    def build(self,):
        self.tacol = TACOL(self.qig, self.mems, self.comms, self.W, None, self.timeout)
        self.tacol.build()
        self.tacol.model.update()

        slack_bound = max(len(self.mems), max(self.mems))
        converter = Gurobi2Qubo(self.tacol.model, slack_bound)
        self.qubo_model: QuboModel = converter.convert()
        print("qubo variables:", len(self.qubo_model.get_variables()))
        print("qubo constraints:", len(self.qubo_model.get_constraints()))


        self.process_num = 16
        self.sa = SimulatedAnnealingOptimizer(
            initial_temperature=100.0,
            alpha=0.99,
            cutoff_temperature=1e-2,
            iterations_per_t=10,
            size_limit=1000,
            process_num=1
            )
        self.controller = kw.common.SolverLoopController(max_repeat_step=10)
        self.solver = kw.solver.PenaltyMethodSolver(self.sa, self.controller)
        # self.solver = kw.solver.SimpleSolver(self.sa)
        # kw.common.CheckpointManager.save_dir = '/tmp/kaiwu_checkpoints/'

    def solve(self,):
        assert self.tacol is not None, "Model not built yet. Call build() first."

        sol_dicts = self.solver.solve_qubo_multi_results(self.qubo_model, size_limit=10000)
        print(len(sol_dicts), "solutions found.")
        for sol_dict in sol_dicts:
            print("Solution dict:", sol_dict)
            val = self.qubo_model.get_value(sol_dict)
            violation, vio_dict = self.qubo_model.verify_constraint(sol_dict)
            print(f"Solution: obj={val}, violation={violation}")

        # print(sol_dicts)

    def get_objs(self):
        return self.solutions

    def get_topology(self):
        return None



if __name__ == "__main__":
    # Example usage
    import numpy as np
    import random

    np.random.seed(42)
    random.seed(42)

    model = gp.Model("example")
    x = model.addVar(vtype=gp.GRB.BINARY, name="x")
    y = model.addVar(vtype=gp.GRB.BINARY, name="y")
    z = model.addVar(vtype=gp.GRB.BINARY, name="z")

    model.addConstr(x + y <= 1, "c0")
    model.addConstr(y + z <= 1, "c1")
    model.addQConstr(x*y + z >= 1, "qc0")

    model.setObjective(x + y + x*y + z, gp.GRB.MINIMIZE)
    model.update()
    # model.optimize()

    converter = Gurobi2Qubo(model, 3)
    qubo_model: QuboModel = converter.convert(penalty=10)
    Q: np.ndarray = qubo_model.get_matrix()

    print(Q)

    # sa = SASolver(process_num=8)
    # results = sa.solve(
    #     model, 
    #     slack_bound=3, 
    #     # penalty_strength=10.0, 
    #     # timeout=10
    #     )
    # # keep the best solution with smallest violation
    # results = sorted(results, key=lambda x: (x[2], x[1]))
    # print(results)

    