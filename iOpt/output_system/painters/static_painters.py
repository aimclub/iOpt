import numpy as np

from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.solution import Solution
from iOpt.output_system.painters.plotters.plotters import Plotter2D, Plotter3D, DisretePlotter
from iOpt.output_system.painters.painter import Painter

import matplotlib.pyplot as plt
import os

class DisretePainter(Painter):
    def __init__(self, searchData, pcount, floatdim, optimumPoint, disreteValues, disreteName, id,
                 mode, subparameters, lb, rb, fileName, pathForSaves, calc):
        self.pathForSaves = pathForSaves
        self.fileName = fileName

        self.mode = mode
        self.id = id

        # формируем массив точек итераций для графика

        self.disretePoints = disreteValues
        self.calculate = calc
        self.optimum = optimumPoint
        self.subparameters = subparameters

        self.values = []
        self.points = []

        self.allValues = []
        self.allPoints = []
        self.Xs = []

        for _ in range(len(disreteValues[id])):
            self.points.append([])
            self.values.append([])

        for item in searchData:
            if item.GetZ() > 1.7e+308:
                continue
            self.Xs.append(item.GetX())
            self.allPoints.append(item.GetY())
            self.allValues.append(item.GetZ())
            disVal = item.GetY().discreteVariables[id]
            disValNum = disreteValues[id].index(disVal)
            self.values[disValNum].append(item.GetZ())

        #настройки графика
        self.plotter = DisretePlotter(self.mode, pcount, floatdim, disreteValues, disreteName, id, self.subparameters, lb, rb)

    def PaintObjectiveFunc(self, numpoints):
        bestcombination = [[], []]
        other = [[], []]
        for x in self.allPoints:
            if x.discreteVariables == self.optimum.discreteVariables:
                bestcombination[0].append(x.floatVariables[self.subparameters[0] - 1])
                bestcombination[1].append(x.floatVariables[self.subparameters[1] - 1])
            else:
                other[0].append(x.floatVariables[self.subparameters[0] - 1])
                other[1].append(x.floatVariables[self.subparameters[1] - 1])
        self.plotter.PlotByGrid(self.CalculateFunc, self.optimum, bestcombination, other, numpoints, 2)

    def PaintPoints(self, currPoint: SearchDataItem = None):
        self.plotter.PlotPoints(self.disretePoints, self.id, self.values, self.allPoints, self.allValues, self.optimum,
                                self.Xs, self.CalculateFunc, 'blue', 'o', 2)

    def PaintOptimum(self, solution: Solution = None):
        pass

    def SaveImage(self):
        if not os.path.isdir(self.pathForSaves):
            if self.pathForSaves == "":
                plt.savefig(self.fileName)
            else:
                os.mkdir(self.pathForSaves)
                plt.savefig(self.pathForSaves + "/" + self.fileName)
        else:
            plt.savefig(self.pathForSaves + "/" + self.fileName)
        plt.show()

    def CalculateFunc(self, x, d):
        point = Point(x, d)
        fv = FunctionValue()
        fv = self.calculate(point, fv)
        return fv.value
class StaticPainter(Painter):
    def __init__(self, searchData: SearchData,
                 solution: Solution,
                 mode,
                 isPointsAtBottom,
                 parameterInNDProblem,
                 pathForSaves,
                 fileName
    ):
        self.pathForSaves = pathForSaves
        self.fileName = fileName

        self.objectFunctionPainterType = mode
        self.isPointsAtBottom = isPointsAtBottom

        self.objFunc = solution.problem.Calculate

        # формируем массив точек итераций для графика
        self.points = []
        self.values = []

        for item in searchData:
            self.points.append(item.GetY().floatVariables[parameterInNDProblem])
            self.values.append(item.GetZ())

        self.points = self.points[1:-1]
        self.values = self.values[1:-1]

        self.optimum = solution.bestTrials[0].point.floatVariables
        self.optimumC = solution.bestTrials[0].point.floatVariables[parameterInNDProblem]
        self.optimumValue = solution.bestTrials[0].functionValues[0].value

        # настройки графика
        self.plotter = Plotter2D(parameterInNDProblem,
                        float(solution.problem.lowerBoundOfFloatVariables[parameterInNDProblem]),
                        float(solution.problem.upperBoundOfFloatVariables[parameterInNDProblem]))

    def PaintObjectiveFunc(self):
        if self.objectFunctionPainterType == 'objective function':
            section = self.optimum.copy()
            self.plotter.PlotByGrid(self.CalculateFunc, section, pointsCount=150)
        elif self.objectFunctionPainterType == 'approximation':
            self.plotter.PlotApproximation(self.points, self.values, pointsCount=100)
        elif self.objectFunctionPainterType == 'interpolation':
            self.plotter.PlotInterpolation(self.points, self.values, pointsCount=100)
        elif self.objectFunctionPainterType == 'only points':
            pass

    def PaintPoints(self, currPoint: SearchDataItem = None):
        if self.isPointsAtBottom:
            values = [self.optimumValue - (max(self.values) - min(self.values)) * 0.3] * len(self.values)
            self.plotter.PlotPoints(self.points, values, 'blue', 'o', 4)
        else:
            self.plotter.PlotPoints(self.points, self.values, 'blue', 'o', 4)

    def PaintOptimum(self, solution: Solution = None):
        value = self.optimumValue
        if self.isPointsAtBottom:
            value = value - (max(self.values) - min(self.values)) * 0.3
        self.plotter.PlotPoints([self.optimumC], [value], 'red', 'o', 4)

    def SaveImage(self):
        if not os.path.isdir(self.pathForSaves):
            if self.pathForSaves == "":
                plt.savefig(self.fileName)
            else:
                os.mkdir(self.pathForSaves)
                plt.savefig(self.pathForSaves + "/" + self.fileName)
        else:
            plt.savefig(self.pathForSaves + "/" + self.fileName)
        plt.show()

    def CalculateFunc(self, x):
        point = Point(x)
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value

class StaticPainterND(Painter):
    def __init__(self, searchData, solution, parameters, mode, calc, fileName, pathForSaves):
        self.pathForSaves = pathForSaves
        self.fileName = fileName

        self.objectFunctionPainterType = mode
        self.objectFunctionCalculatorType = calc

        self.objFunc = solution.problem.Calculate

        # формируем массив точек итераций для графика
        self.points = []
        self.values = []

        for item in searchData:
            self.points.append([item.GetY().floatVariables[parameters[0]], item.GetY().floatVariables[parameters[1]]])
            self.values.append(item.GetZ())

        self.points = self.points[1:-1]
        self.values = self.values[1:-1]

        self.optimum = solution.bestTrials[0].point.floatVariables
        self.optimumValue = solution.bestTrials[0].functionValues[0].value

        self.leftBounds = [float(solution.problem.lowerBoundOfFloatVariables[parameters[0]]),
                           float(solution.problem.lowerBoundOfFloatVariables[parameters[1]])]
        self.rightBounds = [float(solution.problem.upperBoundOfFloatVariables[parameters[0]]),
                            float(solution.problem.upperBoundOfFloatVariables[parameters[1]])]

        # настройки графика
        self.plotter = Plotter3D(parameters, self.leftBounds, self.rightBounds, solution.problem.Calculate, self.objectFunctionPainterType)

    def PaintObjectiveFunc(self):
        if self.objectFunctionPainterType == 'lines layers':
            if self.objectFunctionCalculatorType == 'objective function':
                self.plotter.PlotByGrid(self.CalculateFunc, self.optimum, pointsCount=100)
            elif self.objectFunctionCalculatorType == 'interpolation':
                self.plotter.PlotInterpolation(self.points, self.values, pointsCount=100)
            elif self.objectFunctionCalculatorType == "approximation":
                pass
        elif self.objectFunctionPainterType == 'surface':
            if self.objectFunctionCalculatorType == 'approximation':
                self.plotter.PlotApproximation(self.points, self.values, pointsCount=50)
            elif self.objectFunctionCalculatorType == 'interpolation':
                self.plotter.PlotInterpolation(self.points, self.values, pointsCount=50)
            elif self.objectFunctionCalculatorType == "objective function":
                pass

    def PaintPoints(self, currPoint: SearchDataItem = None):
        self.plotter.PlotPoints(self.points, self.values, 'blue', 'o', 4)

    def PaintOptimum(self, solution: Solution = None):
        self.plotter.PlotPoints([self.optimum], [self.optimumValue], 'red', 'o', 4)

    def SaveImage(self):
        if not os.path.isdir(self.pathForSaves):
            if self.pathForSaves == "":
                plt.savefig(self.fileName)
            else:
                os.mkdir(self.pathForSaves)
                plt.savefig(self.pathForSaves + "/" + self.fileName)
        else:
            plt.savefig(self.pathForSaves + "/" + self.fileName)
        plt.show()

    def CalculateFunc(self, x):
        point = Point(x, [])
        fv = FunctionValue()
        fv = self.objFunc(point, fv)
        return fv.value