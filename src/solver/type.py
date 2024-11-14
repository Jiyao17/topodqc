

from typing import NewType

ProcMemNum = NewType('ProcMemNum', int)
ClusterMem = NewType('ClusterMem', list[ProcMemNum])