import copy
import sys

from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem
from iOpt.trial import FunctionValue, FunctionType


class IndexMethodEvaluate(ICriterionEvaluateMethod):
    """
    The Method class contains an implementation of the Global Search Algorithm
    """

    def __init__(self,
                 task: OptimizationTask
                 ):
        self.task = task

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Perform a search trial at a given point

        :param point: the point at which the trial is to be performed.

        :return: the point at which the trial results are saved.
        """
        try:
            number_of_constraints = self.task.problem.number_of_constraints
            for i in range(number_of_constraints):
                point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)
                point = self.task.calculate(point, i)
                point.set_z(point.function_values[i].value)
                point.set_index(i)
                if point.get_z() > 0:
                    return point
            point.function_values[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV, 0)
            point = self.task.calculate(point, number_of_constraints)
            point.set_z(point.function_values[number_of_constraints].value)
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
        dist_point.set_z(src_point.get_z())
        dist_point.set_index(src_point.get_index())
