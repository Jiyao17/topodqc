
import gurobipy as gp
from src.solver.HQCMCBD import HQCMCBD_algorithm


# Optimize the model
model = gp.Model("Example_model")

# Add variables x, y, z
x = model.addVar(name="x", vartype=gp.GRB.BINARY)
y = model.addVar(name="y", vartype=gp.GRB.BINARY)
z = model.addVar(name="z", vartype=gp.GRB.BINARY)
# set objective function,
objective = x*y*z + x*y - x*z - y*z
model.setObjective(objective, sense=gp.GRB.MAXIMIZE)
# set constraints
model.addConstr(x*y <= z, name="c1")
model.addConstr(x + y == z, name="c2")

model.optimize()
Solver = HQCMCBD_algorithm(model, mode = "manual")
Solver.run()
