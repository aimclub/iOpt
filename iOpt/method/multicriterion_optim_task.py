from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from iOpt.method.optim_task import OptimizationTask, TypeOfCalculation
from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem


class Convolution(ABC):
    """
    Класс Convolution является базовым классом для различных сверток.
    Предполагается, что для каждого набора lambda будет порождён отдельный объект свертки.
    Также возможен вариант, что свертка изменяется по ссылке передаваемой через конструктор класса.
    """

    def __int__(self,
                problem: Problem,
                lambda_param: np.ndarray(shape=(1), dtype=np.double)
                ):
        self.problem = problem
        self.lambda_param = lambda_param

    # Свертка меняет z у SearchDataItem. Z используется в методе для вычисления характеристик
    @abstractmethod
    def calculate_convolution(self,
                              data_item: SearchDataItem,
                              min_value: np.ndarray(shape=(1), dtype=np.double) = [],
                              max_value: np.ndarray(shape=(1), dtype=np.double) = []
                              ) -> SearchDataItem:
        pass


class MulticriterionOptimizationTask(OptimizationTask):
    def __init__(self,
                 problem: Problem,
                 convolution: Convolution,
                 perm: np.ndarray(shape=(1), dtype=int) = None
                 ):
        super().__init__(problem, perm)
        self.convolution = convolution
        # !!! реализовать заполнение массива
        self.min_value: np.ndarray(shape=(1), dtype=np.double) = []
        self.max_value: np.ndarray(shape=(1), dtype=np.double) = []


def calculate(self,
              data_item: SearchDataItem,
              function_index: int,
              calculation_type: TypeOfCalculation = TypeOfCalculation.FUNCTION
              ) -> SearchDataItem:
    """Compute selected function by number."""
    if calculation_type == TypeOfCalculation.FUNCTION:
        data_item.function_values[self.perm[function_index]] = self.problem.calculate(data_item.point,
                                                                                      data_item.function_values[
                                                                                          self.perm[function_index]])
        if not np.isfinite(data_item.function_values[self.perm[function_index]].value):
            raise Exception("Infinity values")

    else:
        # Оновить при реализации метода
        data_item = self.convolution.calculate_convolution(data_item, self.min_value, self.max_value)

    return data_item
