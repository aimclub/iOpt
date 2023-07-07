from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.output_system.painters.painter import Painter
from iOpt.output_system.painters.plotters.plotters import AnimatePlotter2D, AnimatePlotter3D

import matplotlib.pyplot as plt
import os

class AnimatePainter(Painter):
    def __init__(self, isPointsAtBottom, parameterInNDProblem, pathForSaves, fileName):
        self.pathForSaves = pathForSaves
        self.fileName = fileName
        self.isPointsAtBottom = isPointsAtBottom
        self.objFunc = None
        self.parameterInNDProblem = parameterInNDProblem
        self.section = []

        # настройки графика
        self.plotter = AnimatePlotter2D(parameterInNDProblem, 0, 1)

    def SetProblem(self, problem: Problem):
        self.objFunc = problem.calculate

        for i in range(problem.number_of_float_variables):
            self.section.append(float(problem.upper_bound_of_float_variables[i]) - float(problem.lower_bound_of_float_variables[i]))

        # настройки графика
        self.plotter.SetBounds(float(problem.lower_bound_of_float_variables[self.parameterInNDProblem]),
                                 float(problem.upper_bound_of_float_variables[self.parameterInNDProblem]))

    def PaintObjectiveFunc(self):
        self.plotter.PlotByGrid(self.CalculateFunc, self.section.copy(), pointsCount=150)

    def PaintPoints(self, currPoints):
        x = [currPoint.get_y().float_variables[self.parameterInNDProblem] for currPoint in currPoints]
        fv = [currPoint.get_z() for currPoint in currPoints]
        if self.isPointsAtBottom:
            fv = [currPoint.get_z() * 0.7 for currPoint in currPoints]
        else:
            fv = [currPoint.get_z() for currPoint in currPoints]
        self.plotter.PlotPoints(x, fv, 'blue', 'o', 4)

    def PaintOptimum(self, solution: Solution):
        optimum = solution.best_trials[0].point.float_variables[self.parameterInNDProblem]
        optimumVal = solution.best_trials[0].function_values[0].value

        value = optimumVal

        if self.isPointsAtBottom:
            value = value * 0.7

        self.plotter.PlotPoints([optimum], [value], 'red', 'o', 4)

    def SaveImage(self):
        if not os.path.isdir(self.pathForSaves):
            if self.pathForSaves == "":
                plt.savefig(self.fileName)
            else:
                os.mkdir(self.pathForSaves)
                plt.savefig(self.pathForSaves + "/" + self.fileName)
        else:
            plt.savefig(self.pathForSaves + "/" + self.fileName)
        plt.ioff()
        plt.show()

    def CalculateFunc(self, x):
        point = Point(x, [])
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value


class AnimatePainterND(Painter):
    def __init__(self, parametersInNDProblem, pathForSaves, fileName):
        self.pathForSaves = pathForSaves
        self.fileName = fileName
        self.objFunc = None
        self.parametersInNDProblem = parametersInNDProblem
        self.section = []

        # настройки графика
        self.plotter = AnimatePlotter3D(parametersInNDProblem)

    def SetProblem(self, problem: Problem):
        self.objFunc = problem.calculate

        # настройки графика
        self.plotter.SetBounds([float(problem.lower_bound_of_float_variables[self.parametersInNDProblem[0]]),
                               float(problem.lower_bound_of_float_variables[self.parametersInNDProblem[1]])],
                               [float(problem.upper_bound_of_float_variables[self.parametersInNDProblem[0]]),
                               float(problem.upper_bound_of_float_variables[self.parametersInNDProblem[1]])])

    def PaintObjectiveFunc(self):
        self.plotter.PlotByGrid(self.CalculateFunc, self.section, pointsCount=150)

    def PaintPoints(self, currPoints):
        x = [[currPoint.get_y().float_variables[self.parametersInNDProblem[0]] for currPoint in currPoints],
             [currPoint.get_y().float_variables[self.parametersInNDProblem[1]] for currPoint in currPoints]]
        self.plotter.PlotPoints(x, [], 'blue', 'o', 4)

    def PaintOptimum(self, solution: Solution):
        optimum = [solution.best_trials[0].point.float_variables[self.parametersInNDProblem[0]],
                   solution.best_trials[0].point.float_variables[self.parametersInNDProblem[1]]]
        optimumVal = solution.best_trials[0].function_values[0].value

        self.plotter.PlotPoints(optimum, [], 'red', 'o', 4)

        self.section = optimum

    def SaveImage(self):
        if not os.path.isdir(self.pathForSaves):
            if self.pathForSaves == "":
                plt.savefig(self.fileName)
            else:
                os.mkdir(self.pathForSaves)
                plt.savefig(self.pathForSaves + "/" + self.fileName)
        else:
            plt.savefig(self.pathForSaves + "/" + self.fileName)
        plt.ioff()
        plt.show()

    def CalculateFunc(self, x):
        point = Point(x, [])
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value