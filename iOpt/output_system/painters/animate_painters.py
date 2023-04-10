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
        self.objFunc = problem.Calculate

        for i in range(problem.numberOfFloatVariables):
            self.section.append(float(problem.upperBoundOfFloatVariables[i]) - float(problem.lowerBoundOfFloatVariables[i]))

        # настройки графика
        self.plotter.SetBounds(float(problem.lowerBoundOfFloatVariables[self.parameterInNDProblem]),
                                 float(problem.upperBoundOfFloatVariables[self.parameterInNDProblem]))

    def PaintObjectiveFunc(self):
        self.plotter.PlotByGrid(self.CalculateFunc, self.section.copy(), pointsCount=150)

    def PaintPoints(self, currPoint: SearchDataItem):
        x = [currPoint[0].GetY().floatVariables[self.parameterInNDProblem]]
        fv = [currPoint[0].GetZ()]
        if self.isPointsAtBottom:
            fv = [currPoint[0].GetZ() * 0.7]
        else:
            fv = [currPoint[0].GetZ()]
        self.plotter.PlotPoints(x, fv, 'blue', 'o', 4)

    def PaintOptimum(self, solution: Solution):
        optimum = solution.bestTrials[0].point.floatVariables[self.parameterInNDProblem]
        optimumVal = solution.bestTrials[0].functionValues[0].value

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
        self.objFunc = problem.Calculate

        # настройки графика
        self.plotter.SetBounds([float(problem.lowerBoundOfFloatVariables[self.parametersInNDProblem[0]]),
                               float(problem.lowerBoundOfFloatVariables[self.parametersInNDProblem[1]])],
                               [float(problem.upperBoundOfFloatVariables[self.parametersInNDProblem[0]]),
                               float(problem.upperBoundOfFloatVariables[self.parametersInNDProblem[1]])])

    def PaintObjectiveFunc(self):
        self.plotter.PlotByGrid(self.CalculateFunc, self.section, pointsCount=150)

    def PaintPoints(self, savedNewPoints: SearchDataItem):
        x = [savedNewPoints[0].GetY().floatVariables[self.parametersInNDProblem[0]],
        savedNewPoints[0].GetY().floatVariables[self.parametersInNDProblem[1]]]
        self.plotter.PlotPoints(x, [], 'blue', 'o', 4)

    def PaintOptimum(self, solution: Solution):
        optimum = [solution.bestTrials[0].point.floatVariables[self.parametersInNDProblem[0]],
                   solution.bestTrials[0].point.floatVariables[self.parametersInNDProblem[1]]]
        optimumVal = solution.bestTrials[0].functionValues[0].value

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