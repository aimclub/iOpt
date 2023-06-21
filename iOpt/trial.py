import numpy as np
from enum import Enum


class FunctionType(Enum):
    OBJECTIV = 1
    CONSTRAINT = 2


class Point:
    def __init__(self,
                 floatVariables: np.ndarray(shape=(1), dtype=np.double),
                 discreteVariables: np.ndarray(shape=(1), dtype=str)=None,
                 ):
        self.floatVariables = floatVariables
        self.discreteVariables = discreteVariables


class FunctionValue:
    def __init__(self,
                 type: FunctionType = FunctionType.OBJECTIV,
                 functionID: int = 0
                 ):
        self.type = type
        self.functionID = functionID
        self.value: np.double = 0.0


class Trial:
    def __init__(self,
                 point: Point,
                 functionValues: np.ndarray(shape=(1), dtype=FunctionValue)
                 ):
        self.point = point
        self.functionValues = functionValues
