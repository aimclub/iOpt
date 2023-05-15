import copy

from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem
from iOpt.trial import FunctionValue, FunctionType


class IndexMethodCalculator(ICriterionEvaluateMethod):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 task: OptimizationTask
                 ):
        self.task = task

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        number_of_constraints = self.task.problem.numberOfConstraints
        for i in range(number_of_constraints):
            point.functionValues[i] = FunctionValue(FunctionType.CONSTRAINT, i)
            point = self.task.Calculate(point, i)
            point.SetZ(point.functionValues[i].value)
            point.SetIndex(i)
            if point.GetZ() < 0:
                return point
        point.functionValues[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV, 0)
        point = self.task.Calculate(point, number_of_constraints)
        point.SetZ(point.functionValues[number_of_constraints].value)
        point.SetIndex(number_of_constraints)
        return point

    def CopyFunctionals(self, dist_point: SearchDataItem, src_point: SearchDataItem):
        r"""
        Копирование поискового испытания.

        :param dist_point: точка, в которую копируются значения испытаний.
        :param src_point: точка c результатами испытаний.
        """

        dist_point.functionValues = copy.deepcopy(src_point.functionValues)
        dist_point.SetZ(src_point.GetZ())
        dist_point.SetIndex(src_point.GetIndex())
