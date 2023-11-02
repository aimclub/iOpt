from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.local_optimizer import local_optimize
from iOpt.method.method import Method
from iOpt.method.multi_objective_optim_task import MultiObjectiveOptimizationTask, MinMaxConvolution
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType
from iOpt.method.process import Process


class MCOProcess(Process):
    """
    Класс MCOProcess скрывает внутреннюю имплементацию класса Solver.
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener]
                 ):
        super().__init__(parameters, task, evolvent, search_data, method, listeners)
        self.lambdas = [1.0 / task.problem.number_of_objectives] * task.problem.number_of_objectives
        self.convolution = MinMaxConvolution(self.task.problem, self.lambdas)
        self.base_task = task
        self.task = MultiObjectiveOptimizationTask(self.base_task.problem, self.convolution)
        self.method.task = self.task  # метод работает со своей собственной task, поэтому нужно заменить на правильную

    # остальные методы на данном этапе можно вызывать из родительского класса
    # TODO: проверить load/store
