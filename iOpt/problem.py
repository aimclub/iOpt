from abc import abstractmethod
from typing import List

from iOpt.trial import FunctionValue, Point, Trial


class Problem:
    """Base class for optimization problems"""

    def __init__(self) -> None:
        self.numberOfFloatVariables = 0
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 0
        self.numberOfConstraints = 0

        self.floatVariableNames: List[str] = []
        self.discreteVariableNames: List[str] = []

        self.lowerBoundOfFloatVariables: List[float] = []
        self.upperBoundOfFloatVariables: List[float] = []
        self.discreteVariableValues: List[str] = []

        self.knownOptimum: List[Trial] = []

    @abstractmethod
    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Compute selected function at given point.
        For any new problem that inherits from :class:`Problem`, this method should be replaced.
        :return: Calculated function value."""
        pass
