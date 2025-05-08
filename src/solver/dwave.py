import dimod

# formulate a binary cubic problem with quadratic constraints
# min   -xyz + xy + xz - yz - x
# s.t.  x + y >= z
#       x*y <= z

# BPM -> CQM -> BQM -> QUBO -> ISING

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
bqm, invert = dimod.cqm_to_bqm(cqm)

# BQM -> QUBO
# Q, offset = bqm.to_qubo()

# solve the BQM using simulated annealing
from dimod import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=10, label='simulated annealing')

# Print the results
print(sampleset)
# Print the best sample
best_sample = sampleset.first.sample
print("Best sample:", best_sample)