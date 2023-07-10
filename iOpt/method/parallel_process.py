from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.index_method_calculator import IndexMethodCalculator
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solver_parametrs import SolverParameters


class ParallelProcess(Process):
    """
    Класс ParallelProcess реализует распараллеливание на уровне потоков (процессов python).
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener]
                 ):
        """
        Конструктор класса ParallelProcess

        :param parameters: Параметры решения задачи оптимизации.
        :param task: Обёртка решаемой задачи.
        :param evolvent: Развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param search_data: Структура данных для хранения накопленной поисковой информации.
        :param method: Метод оптимизации, проводящий поисковые испытания по заданным правилам.
        :param listeners: Список "наблюдателей" (используется для вывода текущей информации).
        """
        super(ParallelProcess, self).__init__(parameters, task, evolvent, search_data, method, listeners)

        self.index_method_calculator = IndexMethodCalculator(task)
        self.calculator = Calculator(self.index_method_calculator, parameters)

    def do_global_iteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: Количество итераций глобального поиска
        """
        number_ = number
        doneTrials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.before_method_start(self.method)
            doneTrials = self.method.first_iteration(self.calculator)
            self._first_iteration = False
            number = number - 1

        for _ in range(number):
            list_newpoint: list[SearchDataItem] = []
            list_oldpoint: list[SearchDataItem] = []

            for _ in range(self.parameters.number_of_parallel_points):
                newpoint, oldpoint = self.method.calculate_iteration_point()
                list_newpoint.append(newpoint)
                list_oldpoint.append(oldpoint)
            self.calculator.calculate_functionals_for_items(list_newpoint)

            for newpoint, oldpoint in zip(list_newpoint, list_oldpoint):
                self.method.update_optimum(newpoint)
                self.method.renew_search_data(newpoint, oldpoint)
                self.method.finalize_iteration()
                doneTrials = self.search_data.get_last_items(self.parameters.number_of_parallel_points * number_)

        for listener in self._listeners:
            listener.on_end_iteration(doneTrials, self.get_results())
