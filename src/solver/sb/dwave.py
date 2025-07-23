import dimod

# formulate a binary cubic problem with quadratic constraints
# min   -xyz + xy + xz - yz - x
# s.t.  x + y >= z
#       x*y <= z

# BPM -> CQM -> BQM -> QUBO -> ISING
#                        Q  -> SB

# 1. Create BinaryPolynomial objective and constraints
# Cubic Objective: -xyz + xy + xz - yz - x
w, x, y, z = dimod.Binaries(('w', 'x', 'y', 'z'))
poly_obj = {('x', 'y', 'z'): -1, ('x', 'y'): 1, ('x', 'z'): 1, ('y', 'z'): -1, ('x',): -1,}
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
cqm.add_constraint(x + y - z >= 0)
# cqm.add_constraint(x * y - z<= 0)
cqm.add_constraint(w - z <= 0)
cqm.add_constraint(w - x <= 0)
cqm.add_constraint(w - y <= 0)
cqm.add_constraint(w - x - y + 1 >= 0)

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