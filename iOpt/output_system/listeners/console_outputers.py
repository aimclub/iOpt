from iOpt.method.listener import Listener
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method
from iOpt.output_system.outputers.console_outputer import ConsoleOutputer

import numpy as np

class ConsoleOutputListener(Listener):
    """
    Класс ConsoleFullOutputListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на событие консольный вывод.
    """

    def __init__(self, mode='full', iters=100):
        """
        Конструктор класса ConsoleFullOutputListener

        :param mode: Режим вывода в консоль, который будет использован. Возможные режимы: 'full', 'custom' и 'result'.
           Режим 'full' осуществляет в процессе оптимизации полный вывод в консоль получаемой методом поисковой
           информации. Режим 'custom' осуществляет вывод текущей лучшей точки с заданной частотой. Режим 'result'
           выводит в консоль только финальный результат процесса оптимизации.
        :param iters: Частота вывода в консоль. Используется совместно с режимом вывода 'custom'.
        """
        self.__outputer: ConsoleOutputer = None
        self.mode = mode
        self.iters = iters

    def BeforeMethodStart(self, method: Method):
        self.__outputer = ConsoleOutputer(method.task.problem, method.parameters)
        self.__outputer.PrintInitInfo()

    def OnEndIteration(self, currPoint: np.ndarray(shape=(1), dtype=SearchDataItem), currSolution: Solution):
        if self.mode == 'full':
            self.__outputer.PrintIterPointInfo(currPoint)
        elif self.mode == 'custom':
            self.__outputer.PrintBestPointInfo(currSolution, self.iters)
        elif self.mode == 'result':
            pass

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool):
        self.__outputer.PrintFinalResultInfo(solution, status)
