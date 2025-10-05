from abc import ABC, abstractmethod
import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial


class Problem(ABC):
    """Base class for optimization problems"""

    def __init__(self):
        self.name: str = ''
        self.dimension = 0
        self.number_of_float_variables: int = 0
        self.number_of_discrete_variables: int = 0
        self.number_of_objectives: int = 0
        self.number_of_constraints: int = 0

        self.float_variable_names: list[str] = []
        self.discrete_variable_names: list[str] = []

        self.lower_bound_of_float_variables: list[float] = []
        self.upper_bound_of_float_variables: list[float] = []
        self.discrete_variable_values: list[list[str]] = []

        self.known_optimum: list[Trial] = []

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate a function at a given point.
          For any new problem statement that inherits from :class:`Problem`, this method should be overloaded

        :return: Calculated value of the function."""
        function_value.value = 0;
        return function_value

    def calculateAllFunction(self, point: Point, function_values: list[FunctionValue]) -> \
            list[FunctionValue]:
        """
        Calculate all functions at a given point.
          For any new problem statement that inherits from :class:`Problem`, this method should be overloaded

        :return: Calculated values of the functions."""
        for i in range(self.number_of_objectives):
            function_values[i] = self.calculate(point, function_values[i])

        return function_values


    # @abstractmethod
    def get_name(self):
        """
        Get the name of the problem

        :return: self.name."""
        return self.name
        # pass
