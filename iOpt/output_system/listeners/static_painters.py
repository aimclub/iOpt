from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.search_data import SearchData
from iOpt.output_system.painters.static_painters import StaticPainter, StaticPainterND, DiscretePainter, StaticPainterPareto
from iOpt.solution import Solution


class StaticDiscreteListener(Listener):
    """
    """

    def __init__(self, file_name: str, path_for_saves="", mode='analysis', calc='objective function',
                 type='lines layers', numpoints=150, mrkrs=3):
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

        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.subparameters = [1, 2]
        self.mode = mode
        self.type = type
        self.calc = calc
        self.numpoints = numpoints
        self.mrkrs = mrkrs
        self.search_dataSorted = []
        self.bestValueSorted = []
        self.number_of_parallel_points = 1
        self.number_of_constraints = 0

    def before_method_start(self, method: Method):
        self.number_of_constraints = method.task.problem.number_of_constraints
        if method.task.problem.number_of_float_variables > 2 and self.calc == 'interpolation':
            raise Exception(
                "StaticDiscreteListener with calc 'interpolation' supported with dimension <= 2")
        self.number_of_parallel_points = method.parameters.number_of_parallel_points

    def on_end_iteration(self, new_points, solution: Solution):
        for newPoint in new_points:
            self.search_dataSorted.append(newPoint)
            self.bestValueSorted.append(solution.best_trials[0].function_values[self.number_of_constraints].value)

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool):
        painter = DiscretePainter(self.search_dataSorted, self.bestValueSorted,
                                  solution.problem.number_of_discrete_variables,
                                  solution.problem.number_of_float_variables,
                                  solution.best_trials[0].point,
                                  solution.problem.discrete_variable_values,
                                  solution.problem.discrete_variable_names,
                                  self.mode, self.calc, self.subparameters,
                                  solution.problem.lower_bound_of_float_variables,
                                  solution.problem.upper_bound_of_float_variables,
                                  self.file_name, self.path_for_saves, solution.problem.calculate,
                                  solution.best_trials[0].function_values[self.number_of_constraints].value,
                                  search_data, self.number_of_parallel_points, self.number_of_constraints)
        if self.mode == 'analysis':
            painter.paint_analisys(mrks=2)
        elif self.mode == 'bestcombination':
            if self.type == 'lines layers':
                painter.paint_objective_func(self.numpoints)
                painter.paint_points(self.mrkrs)

        painter.save_image()


# mode: objective function, approximation, only points
class StaticPainterListener(Listener):
    """
    The StaticPainterListener class is an event listener. It contains a method handler that produces an image as a reaction to the method's completion.
      as a reaction to the method completion
    """

    def __init__(self, file_name: str, path_for_saves="", indx=0, is_points_at_bottom=False, mode='objective function'):
        """
        Constructor of the StaticPainterListener class

        :param file_name: File name specifying the format for saving the image.
        :param path_for_saves: The directory to save the image. If this parameter is not specified, the image is saved in the current working directory.
           is saved in the current working directory.
        :param indx: Index of the variable of the optimization problem. It is used in multivariate optimization.
           It allows to display in the cross-section of the found minimum the process of optimization by one selected variable.
        :param is_points_at_bottom: Draw search information points below the graph. If False, the points are placed on the graph.
        :param mode: The calculation method for drawing the graph of the objective function that will be used. Possible
           modes: 'objective function', 'only points', 'approximation' and 'interpolation'. The 'objective function' mode
           constructs the graph by calculating the values of the objective function on a uniform grid. The 'approximation' mode builds
           neuroapproximation for the objective function based on the obtained search information.
           The 'interpolation' mode builds interpolation for the objective function based on the obtained search information.
           The 'only points' mode does not plot the objective function.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.parameterInNDProblem = indx
        self.is_points_at_bottom = is_points_at_bottom
        self.mode = mode
        self.number_of_constraints = 0

    def before_method_start(self, method: Method):
        self.number_of_constraints = method.task.problem.number_of_constraints

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool):
        painter = StaticPainter(search_data, solution, self.mode, self.is_points_at_bottom,
                                self.parameterInNDProblem, self.path_for_saves, self.file_name, self.number_of_constraints)
        painter.paint_objective_func()
        painter.paint_points()
        painter.paint_optimum()
        painter.save_image()


# mode: surface, lines layers, approximation
class StaticPainterNDListener(Listener):
    """
    The StaticPainterNDListener class is an event listener. It contains a method handler that produces an image as a
      image as a reaction to the method completion.
      It is used for multidimensional optimization
    """

    def __init__(self, file_name: str, path_for_saves="", vars_indxs=[0, 1], mode='lines layers',
                 calc='objective function'):
        """
        Constructor of the StaticPainterNDListener class

        :param file_name: File name specifying the format for saving the image.
        :param path_for_saves: The directory to save the image. If this parameter is not specified, the image is saved in the current working directory.
           is saved in the current working directory.
        :param vars_indxs: A pair of indices of the variables of the optimization problem for which the figure will be plotted.
        :param mode_: Drawing mode of the objective function graph that will be used.
           Possible modes: 'lines layers', 'surface'.
           The 'lines layers' mode draws level lines in the cross-section of the solution found by the method.
           The 'surface' mode draws the surface in the cross-section of the solution found by the method.
        :param calc: The calculation method for drawing the graph of the objective function that will be used. Possible
           modes: 'objective function' (only in the 'line layers' mode), 'approximation' (only in the 'surface' mode), 'by points'
           and 'interpolation'. The 'objective function' mode builds a graph by calculating the values of the objective function on a uniform grid.
           grid. The 'approximation' mode builds a neuroapproximation for the objective function based on the obtained search information.
           information. The 'interpolation' mode builds interpolation for the objective function based on the obtained search information.
           information.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.parameters = vars_indxs
        self.mode = mode
        self.calc = calc
        self.number_of_constraints = 0
    def before_method_start(self, method: Method):
        self.number_of_constraints = method.task.problem.number_of_constraints

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool, ):
        painter = StaticPainterND(search_data, solution, self.parameters, self.mode, self.calc,
                                  self.file_name, self.path_for_saves, self.number_of_constraints)
        painter.paint_objective_func()
        painter.paint_points()
        painter.paint_optimum()
        painter.save_image()


class StaticPainterParetoListener(Listener):
    """
    The StaticPainterParetoListener class is an event listener. It contains a method handler that produces an image as a
      image as a reaction to the method completion.
      It is used for multicriteria optimization
    """

    def __init__(self, file_name: str, path_for_saves="", criteria_indxs=[0, 1]):
        """
        Constructor of the StaticPainterParetoListener class

        :param file_name: File name specifying the format for saving the image.
        :param path_for_saves: The directory to save the image. If this parameter is not specified, the image is saved in the current working directory.
           is saved in the current working directory.
        :param criteria_indxs: A pair of indices of the criteria of the optimization problem for which the figure will be plotted.
        """
        self.file_name = file_name
        self.path_for_saves = path_for_saves
        self.criteria_indxs = criteria_indxs

    def on_method_stop(self, search_data: SearchData,
                       solution: Solution, status: bool, ):
        painter = StaticPainterPareto(solution, self.criteria_indxs, self.path_for_saves, self.file_name)
        painter.paint_pareto()
        painter.save_image()
