
import torch
import simulated_bifurcation as sb


# Define the tensors
Q = torch.tensor([[8, -17], [4, 0]], dtype=torch.float32)
l = torch.tensor([11, -2], dtype=torch.float32)
c = torch.tensor(5, dtype=torch.float32)  # c = 5 would also work

# Solve using the Simulated Bifurcation algorithm
sb.maximize(Q, l, c, domain="spin") 