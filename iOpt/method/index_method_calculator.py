from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem


class IndexMethodCalculator(ICriterionEvaluateMethod):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 task: OptimizationTask,
                 ):
        self.task = task

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        point = self.task.Calculate(point, 0)
        point.SetZ(point.functionValues[0].value)
        point.SetIndex(0)

        return point

    def CopyFunctionals(self, dist_point: SearchDataItem, src_point: SearchDataItem):
        r"""
        Копирование поискового испытания.

        :param dist_point: точка в которую копиркется значения испытаний.
        :param src_point: точка c результатами испытаний.
        """
        dist_point.functionValues[0] = src_point.functionValues[0]
        dist_point.SetZ(src_point.functionValues[0].value)
        dist_point.SetIndex(0)
