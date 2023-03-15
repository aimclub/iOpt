from __future__ import annotations


from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.method import Method


class IndexMethod(Method):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        super(IndexMethod, self).__init__(parameters, task, evolvent, searchData)

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        pass

    def CalculateM(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление оценки константы Гельдера между между curr_point и left_point.

        :param curr_point: правая точка интервала
        :param left_point: левая точка интервала
        """
        #Обратить внимание на вычисление расстояния, должен использоваться метод CalculateDelta
        pass

    def CalculateGlobalR(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление глобальной характеристики интервала [left_point, curr_point].

        :param curr_point: правая точка интервала.
        :param left_point: левая точка интервала.
        """

        # Сюда переедет целиком CalculateGlobalR из Method, а там останется только случай с равными индексами
        pass

    def UpdateOptimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.

        :param point: точка нового испытания.
        """

        # Сюда переедет целиком UpdateOptimum из Method, а там останется только случай с равными индексами
        pass