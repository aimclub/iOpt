from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.output_system.painters.painter import Painter
from iOpt.output_system.painters.plotters.plotters import AnimatePlotter2D, AnimatePlotter3D

import matplotlib.pyplot as plt
import os

class AnimatePainter(Painter):
    def __init__(self, is_points_at_bottom, parameter_in_nd_problem, path_for_saves, file_name):
        self.path_for_saves = path_for_saves
        self.file_name = file_name
        self.is_points_at_bottom = is_points_at_bottom
        self.objFunc = None
        self.parameterInNDProblem = parameter_in_nd_problem
        self.section = []
        self.number_of_constraints = 0

        # настройки графика
        self.plotter = AnimatePlotter2D(parameter_in_nd_problem, 0, 1)

    def set_problem(self, problem: Problem):
        self.objFunc = problem.calculate
        self.number_of_constraints = problem.number_of_constraints

        for i in range(problem.number_of_float_variables):
            self.section.append(float(problem.upper_bound_of_float_variables[i]) - float(problem.lower_bound_of_float_variables[i]))

        # настройки графика
        self.plotter.set_bounds(float(problem.lower_bound_of_float_variables[self.parameterInNDProblem]),
                                float(problem.upper_bound_of_float_variables[self.parameterInNDProblem]))

    def paint_objective_func(self):
        self.plotter.plot_by_grid(self.calculate_func, self.section.copy(), points_count=150)

    def paint_points(self, curr_points):
        x = [currPoint.get_y().float_variables[self.parameterInNDProblem] for currPoint in curr_points]
        fv = [currPoint.get_z() for currPoint in curr_points]
        if self.is_points_at_bottom:
            fv = [currPoint.get_z() * 0.7 for currPoint in curr_points]
        else:
            fv = [currPoint.get_z() for currPoint in curr_points]
        self.plotter.plot_points(x, fv, 'blue', 'o', 4)

    def paint_optimum(self, solution: Solution):
        optimum = solution.best_trials[0].point.float_variables[self.parameterInNDProblem]
        optimumVal = solution.best_trials[0].function_values[self.number_of_constraints].value

        value = optimumVal

        if self.is_points_at_bottom:
            value = value * 0.7

        self.plotter.plot_points([optimum], [value], 'red', 'o', 4)

    def save_image(self):
        if not os.path.isdir(self.path_for_saves):
            if self.path_for_saves == "":
                plt.savefig(self.file_name)
            else:
                os.mkdir(self.path_for_saves)
                plt.savefig(self.path_for_saves + "/" + self.file_name)
        else:
            plt.savefig(self.path_for_saves + "/" + self.file_name)
        plt.ioff()
        plt.show()

    def calculate_func(self, x):
        point = Point(x, [])
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value


class AnimatePainterND(Painter):
    def __init__(self, parameters_in_nd_problem, path_for_saves, file_name):
        self.path_for_saves = path_for_saves
        self.file_name = file_name
        self.objFunc = None
        self.parametersInNDProblem = parameters_in_nd_problem
        self.section = []
        self.number_of_constraints = 0

        # настройки графика
        self.plotter = AnimatePlotter3D(parameters_in_nd_problem)

    def set_problem(self, problem: Problem):
        self.objFunc = problem.calculate

        # настройки графика
        self.plotter.set_bounds([float(problem.lower_bound_of_float_variables[self.parametersInNDProblem[0]]),
                                 float(problem.lower_bound_of_float_variables[self.parametersInNDProblem[1]])],
                                [float(problem.upper_bound_of_float_variables[self.parametersInNDProblem[0]]),
                               float(problem.upper_bound_of_float_variables[self.parametersInNDProblem[1]])])

    def paint_objective_func(self):
        self.plotter.plot_by_grid(self.calculate_func, self.section, points_count=150)

    def paint_points(self, curr_points):
        x = [[currPoint.get_y().float_variables[self.parametersInNDProblem[0]] for currPoint in curr_points],
             [currPoint.get_y().float_variables[self.parametersInNDProblem[1]] for currPoint in curr_points]]
        self.plotter.plot_points(x, [], 'blue', 'o', 4)

    def paint_optimum(self, solution: Solution):
        optimum = [solution.best_trials[0].point.float_variables[self.parametersInNDProblem[0]],
                   solution.best_trials[0].point.float_variables[self.parametersInNDProblem[1]]]
        optimumVal = solution.best_trials[0].function_values[self.number_of_constraints].value

        self.plotter.plot_points(optimum, [], 'red', 'o', 4)

        self.section = optimum

    def save_image(self):
        if not os.path.isdir(self.path_for_saves):
            if self.path_for_saves == "":
                plt.savefig(self.file_name)
            else:
                os.mkdir(self.path_for_saves)
                plt.savefig(self.path_for_saves + "/" + self.file_name)
        else:
            plt.savefig(self.path_for_saves + "/" + self.file_name)
        plt.ioff()
        plt.show()

    def calculate_func(self, x):
        point = Point(x, [])
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value