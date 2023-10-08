from typing import Tuple

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.multicriterion_optim_task import MulticriterionOptimizationTask
from iOpt.method.search_data import SearchDataItem, SearchData
from iOpt.solver_parametrs import SolverParameters


class MulticriterionMethod(MixedIntegerMethod):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: MulticriterionOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData):
        super().__init__(parameters, task, evolvent, search_data)
        # Флаг используется при
        # 1. Запуске новой задачи (продолжение вычислений с новой сверткой)
        # 2. Обновлении минимума и максимума одного из критериев
        # По влагу необходимо пересчитать все свертки, затем все R и перрезаполнить очередь
        self.is_recalc_all_convolution = True

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        # из IndexMethod
        # Вычисляются ВСЕ критерии
        # Добавить вычисление свертки
        # Желательно использовать одну реализацию из MulticriterionMethodCalculator и не дублировать код
        pass

    def recalc_all_convolution(self) -> None:
        pass

    def calculate_iteration_point(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        #обновление всех сверток по флагу self.is_recalc_all_convolution

        return super(MulticriterionMethod, self).calculate_iteration_point()



    def update_optimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.

        :param point: точка нового испытания.
        """

        # добавить обновление
        # self.task.min_value
        # self.task.max_value
        # из IndexMethod

        # Оптимум поддерживает актуальное состоение найденой области Парето, а не одной точки!

        pass