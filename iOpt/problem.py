from abc import ABC, abstractmethod
import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial


class Problem(ABC):
    """Базовый класс для задач оптимизации"""

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

    @abstractmethod
    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Метод вычисления функции в заданной точке.
          Для любой новой постановки задачи, которая наследуется от :class:`Problem`, этот метод следует перегрузить.

        :return: Вычисленное значение функции."""
        pass

    # @abstractmethod
    def get_name(self):
        """
        Метод позволяет получить имя задачи

        :return: self.name."""
        return self.name
        #pass
