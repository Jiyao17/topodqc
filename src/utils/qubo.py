
import torch

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

