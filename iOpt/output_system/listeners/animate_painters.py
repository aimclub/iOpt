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

    def __init__(self, file_name: str, path_for_saves="", is_points_at_bottom=False, to_paint_obj_func=True):
        """
        Конструктор класса AnimationPaintListener

        :param file_name: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param path_for_saves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param is_points_at_bottom: Должны ли точки поисковой информации ставиться под графиком или нет. Если False,
           точки ставятся на графике.
        :param to_paint_obj_func: Должна ли отрисовываться целевая функция или нет.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.is_points_at_bottom = is_points_at_bottom
        self.to_paint_obj_func = to_paint_obj_func
        self.__painter = AnimatePainter(self.is_points_at_bottom, 0, self.path_for_saves, self.file_name)

    def before_method_start(self, method: Method):
        self.__painter.set_problem(method.task.problem)
        if self.to_paint_obj_func:
            self.__painter.paint_objective_func()

    def on_end_iteration(self, saved_new_points : np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        self.__painter.paint_points(saved_new_points)

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__painter.paint_optimum(solution)
        self.__painter.save_image()

class AnimatePainterNDListener(Listener):
    """
    Класс AnimationPaintListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на события динамически обновляющееся изображение процесса оптимизации.
      Используется для многомерной оптимизации.
    """

    def __init__(self, file_name: str, path_for_saves="", vars_indxs=[0, 1], to_paint_obj_func=True):
        """
        Конструктор класса AnimationNDPaintListener

        :param file_name: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param path_for_saves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param vars_indxs: Пара индексов переменных оптимизационной задачи, для которых будет построен рисунок.
        :param to_paint_obj_func: Должна ли отрисовываться целевая функция или нет.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.to_paint_obj_func = to_paint_obj_func
        self.__painter = AnimatePainterND(vars_indxs, self.path_for_saves, self.file_name)

    def before_method_start(self, method: Method):
        self.__painter.set_problem(method.task.problem)

    def on_end_iteration(self, saved_new_points : np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        self.__painter.paint_points(saved_new_points)

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__painter.paint_optimum(solution)
        if self.to_paint_obj_func:
            self.__painter.paint_objective_func()
        self.__painter.save_image()
