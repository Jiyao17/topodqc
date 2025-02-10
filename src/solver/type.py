

from typing import NewType

ProcMemNum = NewType('ProcMemNum', int)
ProcCommNum = NewType('ProcCommNum', int)
ClusterMem = NewType('ClusterMem', list[ProcMemNum])