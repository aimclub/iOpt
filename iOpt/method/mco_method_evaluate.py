import copy
import sys

from iOpt.method.index_method_evaluate import IndexMethodEvaluate
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem
from iOpt.trial import FunctionValue, FunctionType
from iOpt.method.optim_task import TypeOfCalculation


class MCOMethodEvaluate(IndexMethodEvaluate):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self, task: OptimizationTask):
        super().__init__(task)

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:

        try:
            number_of_constraints = self.task.problem.number_of_constraints
            for i in range(number_of_constraints):
                point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)
                point = self.task.calculate(point, i)
                point.set_z(point.function_values[i].value)
                point.set_index(i)
                if point.get_z() > 0:
                    return point

            for i in range(self.task.problem.number_of_objectives):
                point.function_values[number_of_constraints+i] = FunctionValue(FunctionType.OBJECTIV, i)
                point = self.task.calculate(point, number_of_constraints+i)

            point = self.task.calculate(point, -1, TypeOfCalculation.CONVOLUTION)
            point.set_index(number_of_constraints)

        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-10)

        return point

    def copy_functionals(self, dist_point: SearchDataItem, src_point: SearchDataItem):
        r"""
        Copy the search trial

        :param dist_point: point to which the trial values are copied.
        :param src_point: point with trial results.
        """

        dist_point.function_values = copy.deepcopy(src_point.function_values)
        dist_point.set_index(src_point.get_index())
        if dist_point.get_index() == self.task.problem.number_of_constraints:
            dist_point = self.task.calculate(dist_point, -1, TypeOfCalculation.CONVOLUTION)
