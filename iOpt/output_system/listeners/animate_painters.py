from iOpt.method.listener import Listener
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.output_system.painters.animate_painters import AnimatePainter, AnimatePainterND

import numpy as np


class AnimatePainterListener(Listener):
    """
    The AnimationPaintListener class is an event listener. It contains handler methods that produce a dynamically updating image of the optimization process in response to events.
      dynamically updated image of the optimization process as a reaction to events.
      It is used for one-dimensional optimization
    """

    def __init__(self, file_name: str, path_for_saves="", is_points_at_bottom=False, to_paint_obj_func=True):
        """
        AnimationPaintListener class constructor

        :param file_name: File name specifying the format for saving the image. 
        :param path_for_saves: The directory to save the image. If this parameter is not specified, the image is saved in the current working directory.
           is saved in the current working directory.
        :param is_points_at_bottom: Whether or not the search information points should be placed below the graph. If False,
           the points are placed on the chart.
        :param to_paint_obj_func: Whether or not the objective function should be rendered.
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

    def on_end_iteration(self, saved_new_points: np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        self.__painter.paint_points(saved_new_points)

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__painter.paint_optimum(solution)
        self.__painter.save_image()


class AnimatePainterNDListener(Listener):
    """
    The AnimatePainterNDListener class is an event listener. It contains handler methods that produce a dynamically updating image of the optimization process in response to events.
      dynamically updated image of the optimization process as a reaction to events.
      It is used for multidimensional optimization
    """

    def __init__(self, file_name: str, path_for_saves="", vars_indxs=[0, 1], to_paint_obj_func=True):
        """
        AnimatePainterNDListener class constructor

        :param file_name: File name specifying the format for saving the image. 
        :param path_for_saves: The directory to save the image. If this parameter is not specified, the image is saved in the current working directory.
           is saved in the current working directory.
        :param vars_indxs: A pair of indices of the variables of the optimization problem for which the figure will be plotted.
        :param to_paint_obj_func: Whether the objective function should be rendered or not.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.to_paint_obj_func = to_paint_obj_func
        self.__painter = AnimatePainterND(vars_indxs, self.path_for_saves, self.file_name)

    def before_method_start(self, method: Method):
        self.__painter.set_problem(method.task.problem)

    def on_end_iteration(self, saved_new_points: np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        self.__painter.paint_points(saved_new_points)

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__painter.paint_optimum(solution)
        if self.to_paint_obj_func:
            self.__painter.paint_objective_func()
        self.__painter.save_image()
