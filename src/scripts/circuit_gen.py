

# this file is used to generate quantum circuits in .qasm format

import itertools
import time

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, grover_operator, MCMTGate, HGate, ZGate
from qiskit import qasm2
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import math

import networkx as nx


def qft_gen(num_qubits: int, folder: str):
    qft_circuit = QFT(num_qubits=num_qubits)
    basis_gates = ['u3', 'cx']
    # qft_circuit.decompose(gates_to_decompose=basis_gates)
    # qft_circuit.measure_all()
    start_time = time.time()
    qft_circuit = transpile(qft_circuit, basis_gates=basis_gates, optimization_level=3)
    end_time = time.time()
    print(f"QFT transpile time for {num_qubits} qubits: {end_time - start_time:.2f} seconds")

    filename = folder + f'qft_{num_qubits}.qasm'
    qasm2.dump(qft_circuit, filename)

    print(f"QASM file '{filename}' created successfully.")

# this function is copied from qiskit tutorials
def grover_oracle(marked_states):
    """Build a Grover oracle for multiple marked states
 
    Here we assume all input marked states have the same number of bits
 
    Parameters:
        marked_states (str or list): Marked states of oracle
 
    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle
    """
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    # Compute the number of qubits in circuit
    num_qubits = len(marked_states[0])
 
    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [
            ind
            for ind in range(num_qubits)
            if rev_target.startswith("0", ind)
        ]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if zero_inds:
            qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        if zero_inds:
            qc.x(zero_inds)
    return qc

def grover_gen(num_qubits: int, folder: str):
    # setup the oracle
    # randomly select 1 marked state
    marked_states = np.random.choice([0, 1], size=num_qubits)
    # convert to one single string
    marked_states = [''.join(str(bit) for bit in marked_states), ]
    oracle = grover_oracle(marked_states)
    grover_op = grover_operator(oracle=oracle, insert_barriers=True)
    # optimal_num_iterations = math.floor(math.pi/(4 * math.asin(math.sqrt(len(marked_states) / 2**grover_op.num_qubits))))
    optimal_num_iterations = 1

    # build the Grover circuit
    grover_circuit = QuantumCircuit(grover_op.num_qubits)
    # Create even superposition of all basis states
    grover_circuit.h(range(grover_op.num_qubits))
    print(f"qubits: {grover_op.num_qubits}, marked states: {marked_states}, optimal iterations: {optimal_num_iterations}")
    # Apply Grover operator the optimal number of times
    grover_circuit.compose(grover_op.power(optimal_num_iterations), inplace=True)
    # Measure all qubits
    grover_circuit.measure_all()

    # Implement Grover's algorithm circuit here
    basis_gates = ['u3', 'cx']
    # qft_circuit.decompose(gates_to_decompose=basis_gates)
    # qft_circuit.measure_all()
    grover_circuit = transpile(grover_circuit, basis_gates=basis_gates)

    filename = folder + f'grover_{num_qubits}.qasm'
    qasm2.dump(grover_circuit, filename)

    print(f"QASM file '{filename}' created successfully.")

def qaoa_gen(num_qubits: int, folder: str):
    # a random graph for Max-Cut using Erdos-Renyi model
    graph: nx.Graph = nx.erdos_renyi_graph(n=num_qubits, p=0.5, seed=42)
    # set random weights to edges
    for u, v in graph.edges():
        graph[u][v]['weight'] = np.random.rand() + 0.01  # avoid zero weight
    print("graph generated")
    maxcut = Maxcut(graph)
    print("maxcut instance created")
    op = maxcut.to_quadratic_program()
    qubo = QuadraticProgramToQubo().convert(op)
    cost_op, _ = qubo.to_ising()
    print("cost operator generated")
    qaoa_circuit = QAOAAnsatz(cost_operator=cost_op)

    # bound parameters so it can be transpiled to .qasm
    qaoa_circuit.assign_parameters(
        {param: np.pi/2 for param in qaoa_circuit.parameters}, inplace=True
    )
    basis_gates = ['u3', 'cx']
    qaoa_circuit = transpile(qaoa_circuit, basis_gates=basis_gates)

    filename = folder + f'qaoa_{num_qubits}.qasm'
    qasm2.dump(qaoa_circuit, filename)

    print(f"QASM file '{filename}' created successfully.")


def mcmt_gen(num_qubits: int, folder: str):
    """
    Generate a quantum circuit with a multi-controlled multi-target (MCMT) gate.
    Default gate is a H gate, can be changed in the code below.
    """
    num_cqubits: int = num_qubits // 2
    num_tqubits: int = num_qubits - num_cqubits
    mcmt_circuit = QuantumCircuit(num_cqubits + num_tqubits)
    mcmt_gate = MCMTGate(HGate(), num_cqubits, num_tqubits)
    mcmt_circuit.append(mcmt_gate, range(num_cqubits + num_tqubits))
    mcmt_circuit.measure_all()

    basis_gates = ['u3', 'cx']
    mcmt_circuit = transpile(mcmt_circuit, basis_gates=basis_gates)

    filename = folder + f'mcmt_{num_cqubits}c_{num_tqubits}t.qasm'
    qasm2.dump(mcmt_circuit, filename)

    print(f"QASM file '{filename}' created successfully.")


if __name__ == "__main__":
    circuit_folder = 'src/circuit/src/'
    # sizes = [256, 384, 512, 768, 1024]
    # sizes = [256, 384, 512]
    # sizes = [256*6, 256*8]
    sizes = [1536, ]
    for size in sizes:
        # print(f"Generating circuits of size {size}...")
        mcmt_gen(size, circuit_folder)
        # qft_gen(size, circuit_folder)
        # grover_gen(size, circuit_folder)
