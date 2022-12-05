from enum import Enum
from typing import List, Optional


class FunctionType(Enum):
    OBJECTIV = 1
    CONSTRAINT = 2


class Point:
    def __init__(
        self,
        floatVariables: List[float],
        discreteVariables: Optional[List[str]],
    ):
        self.floatVariables = floatVariables
        self.discreteVariables = discreteVariables


class FunctionValue:
    def __init__(self,
                 type: FunctionType = FunctionType.OBJECTIV,
                 functionID: str = ""
                 ):
        self.type = type
        self.functionID = functionID
        self.value: float = 0.0


class Trial:
    def __init__(
            self,
            point: Point,
            functionValues: list[FunctionValue]
    ):
        self.point = point
        self.functionValues = functionValues
