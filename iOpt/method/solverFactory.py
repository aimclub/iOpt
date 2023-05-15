from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.index_method import IndexMethod
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.parallel_process import ParallelProcess
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters


class SolverFactory:
    """
    Класс SolverFactory создает подходящие классы метода решения и процесса по заданным параметрам
    """

    def __init__(self):
        pass

    @staticmethod
    def CreateMethod(parameters: SolverParameters,
                     task: OptimizationTask,
                     evolvent: Evolvent,
                     searchData: SearchData) -> Method:
        """
        Создает подходящий класс метода решения по заданным параметрам

        :param parameters: параметры решения задачи оптимизации.
        :param task: обёртка решаемой задачи.
        :param evolvent: развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param searchData: структура данных для хранения накопленной поисковой информации.

        :return: созданный метод
        """

        if task.problem.numberOfDisreteVariables > 0:
            return MixedIntegerMethod(parameters, task, evolvent, searchData)
        elif task.problem.numberOfConstraints > 0:
            return IndexMethod(parameters, task, evolvent, searchData)
        else:
            return Method(parameters, task, evolvent, searchData)

    @staticmethod
    def CreateProcess(parameters: SolverParameters,
                      task: OptimizationTask,
                      evolvent: Evolvent,
                      searchData: SearchData,
                      method: Method,
                      listeners: List[Listener]) -> Process:
        """
        Создает подходящий класс процесса по заданным параметрам

        :param parameters: Параметры решения задачи оптимизации.
        :param task: Обёртка решаемой задачи.
        :param evolvent: Развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param searchData: Структура данных для хранения накопленной поисковой информации.
        :param method: Метод оптимизации, проводящий поисковые испытания по заданным правилам.
        :param listeners: Список "наблюдателей" (используется для вывода текущей информации).

        :return: созданный процесс
        """
        if parameters.numberOfParallelPoints == 1:
            return Process(parameters=parameters, task=task, evolvent=evolvent,
                           searchData=searchData, method=method, listeners=listeners)
        else:
            return ParallelProcess(parameters=parameters, task=task, evolvent=evolvent,
                                   searchData=searchData, method=method, listeners=listeners)
