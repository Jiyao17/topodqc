


# Convert Gurobi models to kaiwu QuboModel 
# so it can be solved by kaiwu simulated annealing solver
# kaiwu SA solver offers immediate original objective value

import gurobipy as gp
import kaiwu as kw

from kaiwu.qubo import QuboModel, QuboExpression
from kaiwu.classical import SimulatedAnnealingOptimizer


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
        expr = constr.getAttr("Expr")
        sense = constr.Sense
        rhs = constr.RHS
        name = constr.ConstrName

        # Build the left-hand side expression
        lhs_expr = None
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            if lhs_expr is None:
                lhs_expr = coeff * self.qvars[var.VarName]
            else:
                lhs_expr += coeff * self.qvars[var.VarName]

        # Convert to QuboExpression based on the sense of the constraint
        if sense == gp.GRB.LESS_EQUAL:
            slack = kw.qubo.Integer(f"slack_{name}", 0, self.slack_bound)
            qexpr = lhs_expr - rhs + slack
        elif sense == gp.GRB.GREATER_EQUAL:
            slack = kw.qubo.Integer(f"slack_{name}", 0, self.slack_bound)
            qexpr = rhs - lhs_expr + slack
        elif sense == gp.GRB.EQUAL:
            qexpr = lhs_expr - rhs
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")
        
        return qexpr
        

    def convert(self) -> QuboModel:
        """
        Convert the Gurobi model to a kaiwu QuboModel.
        """
        self._convert_vars()
        obj_expr = self._convert_objective()
        self.qubo_model.set_objective(obj_expr)

        constrs = self.gurobi_model.getConstrs()
        for constr in constrs:
            if isinstance(constr.getAttr("Expr"), gp.LinExpr):
                qconstr = self._convert_linear_constr(constr)
                self.qubo_model.add_constraint(qconstr, constr.ConstrName)


        return self.qubo_model
    

if __name__ == "__main__":
    # Example usage
    model = gp.Model("example")
    x = model.addVar(vtype=gp.GRB.BINARY, name="x")
    y = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=10, name="y")
    z = model.addVar(vtype=gp.GRB.BINARY, name="z")

    model.setObjective(2*x + 3*y + 4*x*y + z, gp.GRB.MINIMIZE)
    model.addConstr(x + y <= 5, "c0")
    model.addConstr(y + z == 7, "c1")

    model.optimize()

    converter = Gurobi2Qubo(model)
    qubo_model = converter.convert()
    # print(qubo_model)

    Q = qubo_model.get_matrix()

    sa = SimulatedAnnealingOptimizer()
    result = sa.solve(qubo_model)
    print("Best solution:", result.best_solution)
    print("Best objective value:", result.best_objective_value)