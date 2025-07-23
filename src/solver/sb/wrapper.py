
import gurobipy as gp

class GurobiModelWrapper:
    """
    A wrapper class to convert a Gurobi model to QUBO format.
    The model should only integer variables.
    """

    def __init__(self, model: 'gp.Model'):
        self.model = model

    def to_qubo(self):
        """
        Convert the Gurobi model to QUBO format.
        """
        pass

    def solve(self):
        """
        Solve the Gurobi model and return the solution.
        """
        pass

    def report(self):
        """
        Report the results of the Gurobi model.
        """
        pass