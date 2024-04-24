from __future__ import annotations

from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

import sys


class DefaultCalculator:
    def __init__(self,
                 evaluate_method: ICriterionEvaluateMethod,
                 parameters: SolverParameters
                 ):
        r"""
        Конструктор класса Calculator

        :param evaluate_method: метод вычислений, проводящий поисковые испытания по заданным правилам.
        :param parameters: параметры решения задачи оптимизации.
        """
        self.evaluate_method = evaluate_method
        self.parameters = parameters

    def calculate_functionals_for_items(self, points: list[SearchDataItem]) -> list[SearchDataItem]:
        r"""
        Метод проведения испытаний для множества точек

        :param points: точки проведения испытаний
        """
        for point in points:
            try:
                self.evaluate_method.calculate_functionals(point)
            except Exception:
                point.set_z(sys.float_info.max)
                point.set_index(-10)

        return points
