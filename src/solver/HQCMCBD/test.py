
import gurobipy as gp
from src.solver.HQCMCBD import HQCMCBD_algorithm

import torch

# formulate a binary cubic problem with quadratic constraints
# min   xyz + xy + xz - yz + x - y
# s.t.  x + y >= z
#       x*z <= y

# Optimize the model
model = gp.Model("Example_model")

# Add variables x, y, z
x = model.addVar(name="x", vtype=gp.GRB.BINARY)
y = model.addVar(name="y", vtype=gp.GRB.BINARY)
xy = model.addVar(name="xy", vtype=gp.GRB.BINARY)
z = model.addVar(name="z", vtype=gp.GRB.BINARY)
xz = model.addVar(name="xz", vtype=gp.GRB.BINARY)
yz = model.addVar(name="yz", vtype=gp.GRB.BINARY)

model.addConstr(xy == x * y, name="xy_definition")  # xy = x * y
model.addConstr(xz == x * z, name="xz_definition")  # xz = x * z
model.addConstr(yz == y * z, name="yz_definition")  # yz = y * z
# set objective function, non linear expression
# w = x*y
objective = xy * z + xy + xz - yz + x - y
model.setObjective(objective, sense=gp.GRB.MAXIMIZE)
# set constraints
model.addConstr(x + y <= z, name="c1")
model.addConstr(x * z <= y, name="c2")
# (x, y, z) = (1, 1, 0), obj = 0 + 1 + 0 - 0 = 1

model.optimize()

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
