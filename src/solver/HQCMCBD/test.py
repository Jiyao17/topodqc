
import gurobipy as gp


# Optimize the model
model = gp.Model("Example_model")
...
model.optimize()
Solver = HQCMCBD_algorithm(model, mode = "manual")
Solver.run()
