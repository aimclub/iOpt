from abc import ABC, abstractmethod
import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial


class Problem(ABC):
    """Base class for optimization problems"""
    def __init__(self):
        self.numberOfFloatVariables: int = 0
        self.numberOfDisreteVariables: int = 0
        self.numberOfObjectives: int = 0
        self.numberOfConstraints: int = 0

        self.floatVariableNames: np.ndarray(shape = (1), dtype = str) = []
        self.discreteVariableNames: np.ndarray(shape = (1), dtype = str) = []

        self.lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = []
        self.upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = []
        self.discreteVariableValues: np.ndarray(shape = (1, 1), dtype = str) = []
        
        self.knownOptimum: np.ndarray(shape = (1), dtype = Trial) = []

    @abstractmethod
    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Compute selected function at given point.
        For any new problem that inherits from :class:`Problem`, this method should be replaced.
        :return: Calculated function value."""
        pass

