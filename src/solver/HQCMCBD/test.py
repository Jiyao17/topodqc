
import gurobipy as gp
from src.solver.HQCMCBD import HQCMCBD_algorithm

import torch


# Optimize the model
model = gp.Model("Example_model")

# Add variables x, y, z
x = model.addVar(name="x", vtype=gp.GRB.BINARY)
y = model.addVar(name="y", vtype=gp.GRB.BINARY)
z = model.addVar(name="z", vtype=gp.GRB.BINARY)
# set objective function, non linear expression
# w = x*y
objective = x*z + y*z
objective = x + y + z
model.setObjective(objective, sense=gp.GRB.MAXIMIZE)
# set constraints
# model.addConstr(x + z <= y, name="c1")
# model.addConstr(x + y >= z, name="c2")
# (x, y, z) = (1, 1, 0), obj = 1*1 + 1*0 - 1*0 = 1

# model.optimize()

PROJ_DIR = '/home/ljy/projects/topodqc/'
config_file = 'src/solver/HQCMCBD/config.json'
abs_path = PROJ_DIR + config_file

# set torch default tensor type to double
torch.set_default_tensor_type(torch.DoubleTensor)
# set all torch tensors to use the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

Solver = HQCMCBD_algorithm(model, mode = "manual", config_file=abs_path)
Solver.run()
