
from typing import NewType



Qubit = NewType('Qubit', int)
Qubits = NewType('Qubits', list[Qubit])

Pair = NewType('Pair', tuple[Qubit, Qubit])
Demand = NewType('Demand', dict[Pair, int])
