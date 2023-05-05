from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.index_method import IndexMethod
from iOpt.method.method import Method
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters


class MethodFactory:
    """
    Класс MethodFactory создает подходящий класс метода решения по заданным параметрам
    """

    def __init__(self):
        pass

    @staticmethod
    def CreateMethod(parameters: SolverParameters,
                     task: OptimizationTask,
                     evolvent: Evolvent,
                     searchData: SearchData) -> Method:
        """
        создает подходящий класс метода решения по заданным параметрам

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
