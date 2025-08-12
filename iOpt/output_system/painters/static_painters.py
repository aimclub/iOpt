import numpy as np

from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.solution import Solution
from iOpt.output_system.painters.plotters.plotters import Plotter2D, Plotter3D, DisretePlotter, PlotterPareto
from iOpt.output_system.painters.painter import Painter

import matplotlib.pyplot as plt
import os

class DiscretePainter(Painter):
    def __init__(self, search_data_sorted, bestsvalues, pcount, floatdim, optimumPoint, discreteValues,
                 discrete_name, mode, calc, subparameters, lb, rb, file_name, path_for_saves, calculate,
                 optimum_value, search_data, number_of_parallel_points, number_of_constraints):
        self.path_for_saves = path_for_saves
        self.file_name = file_name
        self.calc = calc
        self.calculate = calculate
        self.optimum = optimumPoint
        self.optimumVal = optimum_value
        self.number_of_parallel_points = number_of_parallel_points
        self.number_of_constraints = number_of_constraints

        self.values = []
        self.points = []

        self.combination = []

        self.pointsWithBestComb = [[], []]
        self.otherPoints = [[], []]
        self.optimumPoint = [[], []]

        if mode == 'bestcombination':
            for x in search_data:
                if x.get_z() > 1.7e+308:
                    continue
                if x.get_y().discrete_variables != self.optimum.discrete_variables:
                    if floatdim > 1:
                        self.otherPoints[0].append(x.get_y().float_variables[subparameters[0] - 1])
                        self.otherPoints[1].append(x.get_y().float_variables[subparameters[1] - 1])
                    else:
                        self.otherPoints[0].append(x.get_y().float_variables[0])
                        self.otherPoints[1].append(self.optimumVal - 5)
                    continue
                else:
                    if floatdim > 1:
                        '''
                        ok = True
                        for k in range(floatdim):
                            if (x.GetY().float_variables[k] != self.optimum.float_variables[k] and
                            k != subparameters[0] - 1 and k != subparameters[1] - 1):
                                ok = False
                                break
                        if ok:
                            self.values2.append(x.GetZ())
                            self.points2.append([x.GetY().float_variables[subparameters[0] - 1],
                                                 x.GetY().float_variables[subparameters[1] - 1]])
                        '''
                        self.points.append([x.get_y().float_variables[subparameters[0] - 1],
                                            x.get_y().float_variables[subparameters[1] - 1]])
                        self.values.append(x.get_z())
                        self.pointsWithBestComb[0].append(x.get_y().float_variables[subparameters[0] - 1])
                        self.pointsWithBestComb[1].append(x.get_y().float_variables[subparameters[1] - 1])
                    else:
                        self.points.append(x.get_y().float_variables[0])
                        self.values.append(x.get_z())
                        self.pointsWithBestComb[0].append(x.get_y().float_variables[0])
                        self.pointsWithBestComb[1].append(self.optimumVal - 5)

            if floatdim > 1:
                self.optimumPoint[0].append(self.optimum.float_variables[subparameters[0] - 1])
                self.optimumPoint[1].append(self.optimum.float_variables[subparameters[1] - 1])
            else:
                self.optimumPoint[0].append(self.optimum.float_variables[0])
                self.optimumPoint[1].append(self.optimumVal - 5)

        elif mode == 'analysis':
            i = 0
            for item in search_data_sorted:
                i += 1
                if item.get_z() > 1.7e+308:
                    continue
                self.points.append(item.get_y())
                self.values.append([item.get_z(), i])
                str = '['
                for j in range(len(item.get_y().discrete_variables)):
                    str += item.get_y().discrete_variables[j] + ', '
                str = str[:-2]
                str += ']'
                self.combination.append([str, i])

        self.plotter = DisretePlotter(mode, pcount, floatdim, discreteValues, discrete_name,
                                      subparameters, lb, rb, bestsvalues, self.number_of_parallel_points)

    def paint_objective_func(self, numpoints):
        if self.calc == 'objective function':
            section = self.optimum
            self.plotter.plot_by_grid(self.calculate_func, section, numpoints)
        elif self.calc == 'interpolation':
            self.plotter.plot_interpolation(self.points, self.values)

    def paint_points(self, mrks):
        self.plotter.plot_points(self.pointsWithBestComb, self.otherPoints, self.optimum, self.optimumPoint, mrks)

    def paint_analisys(self, mrks):
        self.plotter.plot_analisys_subplots_figure(self.points, self.values, self.combination, self.optimum, mrks)
    def paint_optimum(self, solution: Solution = None):
        pass

    def save_image(self):
        if not os.path.isdir(self.path_for_saves):
            if self.path_for_saves == "":
                plt.savefig(self.file_name)
            else:
                os.mkdir(self.path_for_saves)
                plt.savefig(self.path_for_saves + "/" + self.file_name)
        else:
            plt.savefig(self.path_for_saves + "/" + self.file_name)
        plt.show()

    def calculate_func(self, x, d):
        point = Point(x, d)
        fv = FunctionValue()
        fv = self.calculate(point, fv)
        return fv.value
class StaticPainter(Painter):
    def __init__(self, search_data: SearchData,
                 solution: Solution,
                 mode,
                 is_points_at_bottom,
                 parameter_in_nd_problem,
                 path_for_saves,
                 file_name,
                 number_of_constraints
                 ):
        self.path_for_saves = path_for_saves
        self.file_name = file_name

        self.objectFunctionPainterType = mode
        self.is_points_at_bottom = is_points_at_bottom

        self.objFunc = solution.problem.calculate
        self.number_of_constraints = number_of_constraints

        # формируем массив точек итераций для графика
        self.points = []
        self.values = []

        for item in search_data:
            self.points.append(item.get_y().float_variables[parameter_in_nd_problem])
            self.values.append(item.get_z())

        self.points = self.points[1:-1]
        self.values = self.values[1:-1]

        self.optimum = solution.best_trials[0].point.float_variables
        self.optimumC = solution.best_trials[0].point.float_variables[parameter_in_nd_problem]
        self.optimumValue = solution.best_trials[0].function_values[self.number_of_constraints].value

        # настройки графика
        self.plotter = Plotter2D(parameter_in_nd_problem,
                                 float(solution.problem.lower_bound_of_float_variables[parameter_in_nd_problem]),
                                 float(solution.problem.upper_bound_of_float_variables[parameter_in_nd_problem]))

    def paint_objective_func(self):
        if self.objectFunctionPainterType == 'objective function':
            section = self.optimum.copy()
            self.plotter.plot_by_grid(self.calculate_func, section, points_count=150)
        elif self.objectFunctionPainterType == 'approximation':
            self.plotter.plot_approximation(self.points, self.values, points_count=100)
        elif self.objectFunctionPainterType == 'interpolation':
            self.plotter.plot_interpolation(self.points, self.values, points_count=100)
        elif self.objectFunctionPainterType == 'only points':
            pass

    def paint_points(self, curr_point: SearchDataItem = None):
        if self.is_points_at_bottom:
            values = [self.optimumValue - (max(self.values) - min(self.values)) * 0.3] * len(self.values)
            self.plotter.plot_points(self.points, values, 'blue', 'o', 4)
        else:
            self.plotter.plot_points(self.points, self.values, 'blue', 'o', 4)

    def paint_optimum(self, solution: Solution = None):
        value = self.optimumValue
        if self.is_points_at_bottom:
            value = value - (max(self.values) - min(self.values)) * 0.3
        self.plotter.plot_points([self.optimumC], [value], 'red', 'o', 4)

    def save_image(self):
        if not os.path.isdir(self.path_for_saves):
            if self.path_for_saves == "":
                plt.savefig(self.file_name)
            else:
                os.mkdir(self.path_for_saves)
                plt.savefig(self.path_for_saves + "/" + self.file_name)
        else:
            plt.savefig(self.path_for_saves + "/" + self.file_name)
        plt.show()

    def calculate_func(self, x):
        point = Point(x)
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value

class StaticPainterND(Painter):
    def __init__(self, search_data, solution, parameters, mode, calc, file_name, path_for_saves, number_of_constraints):
        self.path_for_saves = path_for_saves
        self.file_name = file_name

        self.objectFunctionPainterType = mode
        self.objectFunctionCalculatorType = calc

        self.objFunc = solution.problem.calculate
        self.number_of_constraints = number_of_constraints

        # формируем массив точек итераций для графика
        self.points = []
        self.values = []

        for item in search_data:
            self.points.append([item.get_y().float_variables[parameters[0]], item.get_y().float_variables[parameters[1]]])
            self.values.append(item.get_z())

        self.points = self.points[1:-1]
        self.values = self.values[1:-1]

        self.optimum = solution.best_trials[0].point.float_variables
        self.optimum_section = [solution.best_trials[0].point.float_variables[parameters[0]],
                        solution.best_trials[0].point.float_variables[parameters[1]]]

        self.optimumValue = solution.best_trials[0].function_values[self.number_of_constraints].value

        self.leftBounds = [float(solution.problem.lower_bound_of_float_variables[parameters[0]]),
                           float(solution.problem.lower_bound_of_float_variables[parameters[1]])]
        self.rightBounds = [float(solution.problem.upper_bound_of_float_variables[parameters[0]]),
                            float(solution.problem.upper_bound_of_float_variables[parameters[1]])]

        # настройки графика
        self.plotter = Plotter3D(parameters, self.leftBounds, self.rightBounds, solution.problem.calculate,
                                 self.objectFunctionPainterType, self.objectFunctionCalculatorType)

    def paint_objective_func(self):
        if self.objectFunctionPainterType == 'lines layers':
            if self.objectFunctionCalculatorType == 'objective function':
                self.plotter.plot_by_grid(self.calculate_func, self.optimum, points_count=100)
            elif self.objectFunctionCalculatorType == 'interpolation':
                self.plotter.plot_interpolation(self.points, self.values, points_count=100)
            elif self.objectFunctionCalculatorType == 'by points':
               self.plotter.plot_by_points(self.points, self.values)
            elif self.objectFunctionCalculatorType == "approximation":
                pass
        elif self.objectFunctionPainterType == 'surface':
            if self.objectFunctionCalculatorType == 'approximation':
                self.plotter.plot_approximation(self.points, self.values, points_count=50)
            elif self.objectFunctionCalculatorType == 'interpolation':
                self.plotter.plot_interpolation(self.points, self.values, points_count=50)
            elif self.objectFunctionCalculatorType == 'by points':
               self.plotter.plot_by_points(self.points, self.values)
            elif self.objectFunctionCalculatorType == "objective function":
                pass

    def paint_points(self, curr_point: SearchDataItem = None):
        self.plotter.plot_points(self.points, self.values, 'blue', 'o', 4)

    def paint_optimum(self, solution: Solution = None):
        self.plotter.plot_points([self.optimum_section], [self.optimumValue], 'red', 'o', 4)

    def save_image(self):
        if not os.path.isdir(self.path_for_saves):
            if self.path_for_saves == "":
                plt.savefig(self.file_name)
            else:
                os.mkdir(self.path_for_saves)
                plt.savefig(self.path_for_saves + "/" + self.file_name)
        else:
            plt.savefig(self.path_for_saves + "/" + self.file_name)
        plt.show()

    def calculate_func(self, x):
        point = Point(x, [])
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value


class StaticPainterPareto:
    def __init__(self,
                 solution: Solution,
                 criteria_indxs,
                 path_for_saves,
                 file_name
                 ):
        self.path_for_saves = path_for_saves
        self.file_name = file_name

        # numbers of criteria selected by user
        self.first_criteria_indx = criteria_indxs[0]
        self.second_criteria_indx = criteria_indxs[1]

        # values of Pareto-efficient criteria with input indices
        self.first_criteria_values = [trial.function_values[self.first_criteria_indx].value for trial in solution.best_trials]
        self.second_criteria_values = [trial.function_values[self.second_criteria_indx].value for trial in solution.best_trials]

        # definition of plotter
        self.plotter = PlotterPareto()

    def paint_pareto(self):
        self.plotter.plot_pareto(self.first_criteria_values, self.second_criteria_values,
                                 self.first_criteria_indx, self.second_criteria_indx)

    def save_image(self):
        if not os.path.isdir(self.path_for_saves):
            if self.path_for_saves == "":
                plt.savefig(self.file_name)
            else:
                os.mkdir(self.path_for_saves)
                plt.savefig(self.path_for_saves + "/" + self.file_name)
        else:
            plt.savefig(self.path_for_saves + "/" + self.file_name)
        plt.show()
