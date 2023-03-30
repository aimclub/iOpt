from __future__ import annotations
from typing import List

import copy
from typing import Tuple

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

from iOpt.method.method import Method
from iOpt.method.index_method import IndexMethod



class MixedIntegerMethod(IndexMethod):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        super(MixedIntegerMethod, self).__init__(parameters, task, evolvent, searchData)
        numberOfParameterCombinations : int = 0 # определяем количество сочетаний параметров
        arraySearchData : List[SearchData] = [] #локальные копии, в количестве numberOfParameterCombinations
        # в searchData будет общий список точек (для listeners).
        arrayMethod: List[IndexMethod] = [] #локальные копии, в количестве numberOfParameterCombinations со своими SearchData из arraySearchData
        # 0 0.5 1  1.5 2   2.5  3    3.5 4

    def FirstIteration(self) -> None:
        r"""
        Метод выполняет первую итерацию Алгоритма Глобального Поиска.
        """
        # обход всех arrayMethod
        pass

    def CalculateDelta(lPoint: SearchDataItem, rPoint: SearchDataItem, dimension: int) -> float:
        """
        Вычисляет гельдерово расстояние в метрике Гельдера между двумя точками на отрезке [0,1],
          полученными при редукции размерности.

        :param lx: левая точка
        :param rx: правая точка
        :param dimension: размерность исходного пространства

        :return: гельдерово расстояние между lx и rx.
        """
        # Учесть что у левой точки может быть x = 1 и отрицательный индекс, тогда считать что x = 0
        pass;

    def CalculateNextPointCoordinate(self, point: SearchDataItem) -> float:
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}` в заданном интервале :math:`[x_{t-1},x_t]`.

        :param point: интервал, заданный его правой точкой :math:`x_t`.

        :return: точка нового испытания :math:`x^{k+1}` в этом интервале.
        """

        # 0 0.5 1  1.5 2   2.5  3    3.5 4
        pass

    def CalculateIterationPoint(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}`.

        :return: :math:`x^{k+1}` - точка нового испытания, и :math:`x_t` - левая точка интервала :math:`[x_{t-1},x_t]`,
          которому принадлежит :math:`x^{k+1}`, т.е. :math:`x^{k+1} \in [x_{t-1},x_t]`.
        """
        pass