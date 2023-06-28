from iOpt.method.listener import Listener
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.output_system.painters.static_painters import StaticPainter, StaticPainterND, DiscretePainter

import numpy as np

class StaticDiscreteListener(Listener):
    """
    """
    def __init__(self, fileName: str, pathForSaves="", mode='analysis', calc='objective function', type='lines layers',
                 numpoints=150, mrkrs=3):
        """
        """
        if mode != 'analysis' and mode != 'bestcombination':
            raise Exception(
                "StaticDiscreteListener mode is incorrect, mode can take values 'analysis' or 'bestcombination'")
        if type != 'lines layers':
            raise Exception(
                "StaticDiscreteListener type is incorrect, type can take values 'lines layers'")
        if calc != 'objective function' and calc != 'interpolation':
            raise Exception(
                "StaticDiscreteListener calc is incorrect, calc can take values 'objective function' or 'interpolation'")
        if numpoints <= 0:
            raise Exception(
                "StaticDiscreteListener numpoints is incorrect, numpoints > 0")
        if mrkrs <= 0:
            raise Exception(
                "StaticDiscreteListener mrkrs is incorrect, mrkrs > 0")

        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.subparameters = [1, 2]
        self.mode = mode
        self.type = type
        self.calc = calc
        self.numpoints = numpoints
        self.mrkrs = mrkrs
        self.searchDataSorted = []
        self.bestValueSorted = []
        self.numberOfParallelPoints = 1

    def BeforeMethodStart(self, method: Method):
        if method.task.problem.numberOfFloatVariables > 2 and self.calc == 'interpolation':
            raise Exception(
                "StaticDiscreteListener with calc 'interpolation' supported with dimension <= 2")
        self.numberOfParallelPoints = method.parameters.numberOfParallelPoints

    def OnEndIteration(self, newPoints, solution: Solution):
        for newPoint in newPoints:
            self.searchDataSorted.append(newPoint)
            self.bestValueSorted.append(solution.bestTrials[0].functionValues[0].value)

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool):
        painter = DiscretePainter(self.searchDataSorted, self.bestValueSorted,
                                  solution.problem.numberOfDiscreteVariables,
                                  solution.problem.numberOfFloatVariables,
                                  solution.bestTrials[0].point,
                                  solution.problem.discreteVariableValues,
                                  solution.problem.discreteVariableNames,
                                  self.mode, self.calc, self.subparameters,
                                  solution.problem.lowerBoundOfFloatVariables,
                                  solution.problem.upperBoundOfFloatVariables,
                                  self.fileName, self.pathForSaves, solution.problem.Calculate,
                                  solution.bestTrials[0].functionValues[0].value,
                                  searchData, self.numberOfParallelPoints)
        if self.mode == 'analysis':
            painter.PaintAnalisys(mrks=2)
        elif self.mode == 'bestcombination':
            if self.type == 'lines layers':
                painter.PaintObjectiveFunc(self.numpoints)
                painter.PaintPoints(self.mrkrs)

        painter.SaveImage()

# mode: objective function, approximation, only points
class StaticPainterListener(Listener):
    """
    Класс StaticPaintListener - слушатель событий. Содержит метод-обработчик, выдающий в качестве
      реакции на завершение работы метода изображение.
    """

    def __init__(self, fileName: str, pathForSaves="", indx=0, isPointsAtBottom=False, mode='objective function'):
        """
        Конструктор класса StaticPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param indx: Индекс переменной оптимизационной задачи. Используется в многомерной оптимизации.
           Позволяет отобразить в сечении найденного минимума процесс оптимизации по одной выбранной переменной.
        :param isPointsAtBottom: Отрисовать точки поисковой информации под графиком. Если False, точки ставятся на графике.
        :param mode: Способ вычислений для отрисовки графика целевой функции, который будет использован. Возможные
           режимы: 'objective function', 'only points', 'approximation' и 'interpolation'. Режим 'objective function'
           строит график, вычисляя значения целевой функции на равномерной сетке. Режим 'approximation' строит
           нейроаппроксимацию для целевой функции на основе полученной поисковой информации.
           Режим 'interpolation' строит интерполяцию для целевой функции на основе полученной поисковой информации.
           Режим 'only points' не строит график целевой функции.
        """
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameterInNDProblem = indx
        self.isPointsAtBottom = isPointsAtBottom
        self.mode = mode

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool):
        painter = StaticPainter(searchData, solution, self.mode, self.isPointsAtBottom,
                           self.parameterInNDProblem, self.pathForSaves, self.fileName)
        painter.PaintObjectiveFunc()
        painter.PaintPoints()
        painter.PaintOptimum()
        painter.SaveImage()

# mode: surface, lines layers, approximation
class StaticPainterNDListener(Listener):
    """
    Класс StaticNDPaintListener - слушатель событий. Содержит метод-обработчик, выдающий в качестве
      реакции на завершение работы метода изображение.
      Используется для многомерной оптимизации.
    """

    def __init__(self, fileName: str, pathForSaves="", varsIndxs=[0, 1], mode='lines layers',
                 calc='objective function'):
        """
        Конструктор класса StaticNDPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param varsIndxs: Пара индексов переменных оптимизационной задачи, для которых будет построен рисунок.
        :param mode_: Режим отрисовки графика целевой функции, который будет использован.
           Возможные режимы:'lines layers', 'surface'.
           Режим 'lines layers' рисует линии уровня в сечении найденного методом решения.
           Режим 'surface' строит поверхность в сечении найденного методом решения.
        :param calc: Способ вычислений для отрисовки графика целевой функции, который будет использован. Возможные
           режимы: 'objective function' (только в режиме 'lines layers'), 'approximation' (только в режиме 'surface')
           и 'interpolation'. Режим 'objective function' строит график, вычисляя значения целевой функции на равномерной
           сетке. Режим 'approximation' строит нейроаппроксимацию для целевой функции на основе полученной поисковой
           информации. Режим 'interpolation' строит интерполяцию для целевой функции на основе полученной поисковой
           информации.
        """
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.mode = mode
        self.calc = calc

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool, ):
        painter = StaticPainterND(searchData, solution, self.parameters, self.mode, self.calc,
                           self.fileName, self.pathForSaves)
        painter.PaintObjectiveFunc()
        painter.PaintPoints()
        painter.PaintOptimum()
        painter.SaveImage()
