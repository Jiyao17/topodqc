
import torch

import numpy as np
from src.circuit.qig import QIG, RandomQIG

from src.solver.taco_l import TACOL
from src.solver.HQCMCBD import HQCMCBD_algorithm
from src.solver.sb.converter import Model2QUBO



class TACOL_SB(TACOL):
    """"
    Solve the TACOL problem using the SB (simulated bifurcation) method.
    """

    def __init__(self, qig, mems, comms, W, timeout=600):
        super().__init__(qig, mems, comms, W, timeout)
        # self.name = "TACOL_SB"

    def solve_sb(self, config_file, max_steps=1e4):
        self.model.update()
        
        # show variable number
        print("Variable number:", self.model.num_vars)

        solver = Model2QUBO(self.model, mode = "manual", config_file=config_file)
        vec, val, obj = solver.run(max_steps=max_steps)

        print("SB solution vector:", len(vec))
        print("SB solution value:", val)
        print("SB objective value:", obj)


if __name__ == "__main__":

    PROJ_DIR = '/home/ljy/projects/topodqc/'
    config_file = 'src/solver/sb/config.json'
    config_file = PROJ_DIR + config_file

    # set torch default tensor type to double
    torch.set_default_tensor_type(torch.DoubleTensor)
    # set all torch tensors to use the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)


    np.random.seed(0)
    proc_num = 4
    mem = 4
    comm = 4

    # qubit_num = 16
    # demand_pair = int(qubit_num * (qubit_num-1) / 2) # max
    # demand_pair = qubit_num * 2 # moderate

    # qig = RandomQIG(qubit_num, demand_pair, (1, 11))
    qig = QIG.from_qasm('src/circuit/src/0410184_169.qasm')
    # qig.contract(4, inplace=True)

    mems = [mem] * proc_num
    comms = [comm] * proc_num
    W = proc_num * (proc_num-1) / 2
    # W = (proc_num - 1)
    # W = proc_num * 2

    model = TACOL_SB(qig, mems, comms, W)
    model.build()

    # model.solve()

    model.solve_sb(config_file, max_steps=1e4)