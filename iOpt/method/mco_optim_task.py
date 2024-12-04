from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import copy

from iOpt.method.optim_task import OptimizationTask, TypeOfCalculation
from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem


class Convolution(ABC):
    """
    The Convolution class is the base class for various convolutions.
    It is expected that a separate convolution object will be spawned for each set of lambdas.
    It is also possible that the convolution is changed by a reference passed through the class constructor.
    """

    def __init__(self,
                 problem: Problem,
                 lambda_param: np.ndarray(shape=(1), dtype=np.double)
                 ):
        self.problem = problem
        self.lambda_param = lambda_param

    @abstractmethod
    def calculate_convolution(self,
                              data_item: SearchDataItem,
                              min_value: np.ndarray(shape=(1), dtype=np.double) = [],
                              max_value: np.ndarray(shape=(1), dtype=np.double) = []
                              ) -> SearchDataItem:
        pass


class MinMaxConvolution(Convolution):
    """
    minimax convolution
    """

    def __init__(self,
                 problem: Problem,
                 lambda_param: np.ndarray(shape=(1), dtype=np.double),
                 is_scaling: False
                 ):
        self.is_scaling = is_scaling
        super().__init__(problem, lambda_param)

    def calculate_convolution(self,
                              data_item: SearchDataItem,
                              min_value: np.ndarray(shape=(1), dtype=np.double) = [],
                              max_value: np.ndarray(shape=(1), dtype=np.double) = []
                              ) -> SearchDataItem:

        value = -10
        for i in range(0, self.problem.number_of_objectives):
            dx = 1
            if self.is_scaling:
                dx = np.double(max_value[i] - min_value[i])
                if dx < 1e-6:
                    dx = 1

            f_value = (data_item.function_values[i].value - min_value[i]) / dx
            value = max(value, f_value * self.lambda_param[i])

        data_item.set_z(value)
        return data_item


class MCOOptimizationTask(OptimizationTask):
    def __init__(self,
                 problem: Problem,
                 convolution: Convolution,
                 perm: np.ndarray(shape=(1), dtype=int) = None
                 ):
        super().__init__(problem, perm)
        self.convolution = convolution
        self.min_value = np.ndarray(shape=(problem.number_of_objectives,), dtype=np.double)
        self.min_value.fill(0)
        self.max_value = np.ndarray(shape=(problem.number_of_objectives,), dtype=np.double)
        self.max_value.fill(0)
        if self.problem.known_optimum:
            self.min_value = copy.deepcopy(self.problem.known_optimum[0].function_values)
            for know_optimum in self.problem.known_optimum:
                for i in range(0, self.problem.number_of_objectives):
                    if self.min_value[i] > know_optimum.function_values[i]:
                        self.min_value[i] = know_optimum.function_values[i]

    def calculate(self,
                  data_item: SearchDataItem,
                  function_index: int,
                  calculation_type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                  ) -> SearchDataItem:
        """Compute selected function by number."""
        if calculation_type == TypeOfCalculation.FUNCTION:
            if function_index == -1:  # Calculate all criteria
                data_item.function_values = self.problem.calculateAllFunction(data_item.point,
                                                                              data_item.function_values)
                for i in range(self.problem.number_of_objectives):
                    if not np.isfinite(data_item.function_values[self.perm[self.problem.number_of_constraints +
                                                                           i]].value):
                        raise Exception("Infinity values")
            else:
                data_item.function_values[self.perm[function_index]] = \
                    self.problem.calculate(data_item.point, data_item.function_values[self.perm[function_index]])
                if not np.isfinite(data_item.function_values[self.perm[function_index]].value):
                    raise Exception("Infinity values")

        elif calculation_type == TypeOfCalculation.CONVOLUTION:
            data_item = self.convolution.calculate_convolution(data_item, self.min_value, self.max_value)

        return data_item
