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

        self.float_variable_names: np.ndarray(shape=(1), dtype=str) = []
        self.discrete_variable_names: np.ndarray(shape=(1), dtype=str) = []

        self.lower_bound_of_float_variables: np.ndarray(shape=(1), dtype=np.double) = []
        self.upper_bound_of_float_variables: np.ndarray(shape=(1), dtype=np.double) = []
        self.discrete_variable_values: np.ndarray(shape=(1, 1), dtype=str) = []

        self.known_optimum: np.ndarray(shape=(1), dtype=Trial) = []

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate a function at a given point.
          For any new problem statement that inherits from :class:`Problem`, this method should be overloaded

        :return: Calculated value of the function."""
        function_value.value = 0;
        return function_value

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)) -> \
            np.ndarray(shape=(1), dtype=FunctionValue):
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
