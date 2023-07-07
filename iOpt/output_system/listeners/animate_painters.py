from iOpt.method.listener import Listener
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.output_system.painters.animate_painters import AnimatePainter, AnimatePainterND

import numpy as np

class AnimatePainterListener(Listener):
    """
    Класс AnimationPaintListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на события динамически обновляющееся изображение процесса оптимизации.
      Используется для одномерной оптимизации.
    """

    def __init__(self, fileName: str, pathForSaves="", isPointsAtBottom=False, toPaintObjFunc=True):
        """
        Конструктор класса AnimationPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param isPointsAtBottom: Должны ли точки поисковой информации ставиться под графиком или нет. Если False,
           точки ставятся на графике.
        :param toPaintObjFunc: Должна ли отрисовываться целевая функция или нет.
        """
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.isPointsAtBottom = isPointsAtBottom
        self.toPaintObjFunc = toPaintObjFunc
        self.__painter = AnimatePainter(self.isPointsAtBottom, 0, self.pathForSaves, self.fileName)

    def before_method_start(self, method: Method):
        self.__painter.SetProblem(method.task.problem)
        if self.toPaintObjFunc:
            self.__painter.PaintObjectiveFunc()

    def on_end_iteration(self, savedNewPoints : np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        self.__painter.PaintPoints(savedNewPoints)

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__painter.PaintOptimum(solution)
        self.__painter.SaveImage()

class AnimatePainterNDListener(Listener):
    """
    Класс AnimationPaintListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на события динамически обновляющееся изображение процесса оптимизации.
      Используется для многомерной оптимизации.
    """

    def __init__(self, fileName: str, pathForSaves="", varsIndxs=[0, 1], toPaintObjFunc=True):
        """
        Конструктор класса AnimationNDPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param varsIndxs: Пара индексов переменных оптимизационной задачи, для которых будет построен рисунок.
        :param toPaintObjFunc: Должна ли отрисовываться целевая функция или нет.
        """
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.toPaintObjFunc = toPaintObjFunc
        self.__painter = AnimatePainterND(varsIndxs, self.pathForSaves, self.fileName)

    def before_method_start(self, method: Method):
        self.__painter.SetProblem(method.task.problem)

    def on_end_iteration(self, savedNewPoints : np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        self.__painter.PaintPoints(savedNewPoints)

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__painter.PaintOptimum(solution)
        if self.toPaintObjFunc:
            self.__painter.PaintObjectiveFunc()
        self.__painter.SaveImage()
