import numpy as np

from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.solution import Solution
from iOpt.output_system.painters.plotters.plotters import Plotter2D, Plotter3D, DisretePlotter
from iOpt.output_system.painters.painter import Painter

import matplotlib.pyplot as plt
import os

class DiscretePainter(Painter):
    def __init__(self, searchDataSorted, bestsvalues, pcount, floatdim, optimumPoint, discreteValues,
                 discreteName, mode, calc, subparameters, lb, rb, fileName, pathForSaves, calculate, optimumValue, searchData, numberOfParallelPoints):
        self.pathForSaves = pathForSaves
        self.fileName = fileName
        self.calc = calc
        self.calculate = calculate
        self.optimum = optimumPoint
        self.optimumVal = optimumValue
        self.numberOfParallelPoints = numberOfParallelPoints

        self.values = []
        self.points = []

        self.combination = []

        self.pointsWithBestComb = [[], []]
        self.otherPoints = [[], []]
        self.optimumPoint = [[], []]

        if mode == 'bestcombination':
            for x in searchData:
                if x.GetZ() > 1.7e+308:
                    continue
                if x.GetY().discreteVariables != self.optimum.discreteVariables:
                    if floatdim > 1:
                        self.otherPoints[0].append(x.GetY().floatVariables[subparameters[0] - 1])
                        self.otherPoints[1].append(x.GetY().floatVariables[subparameters[1] - 1])
                    else:
                        self.otherPoints[0].append(x.GetY().floatVariables[0])
                        self.otherPoints[1].append(self.optimumVal - 5)
                    continue
                else:
                    if floatdim > 1:
                        '''
                        ok = True
                        for k in range(floatdim):
                            if (x.GetY().floatVariables[k] != self.optimum.floatVariables[k] and
                            k != subparameters[0] - 1 and k != subparameters[1] - 1):
                                ok = False
                                break
                        if ok:
                            self.values2.append(x.GetZ())
                            self.points2.append([x.GetY().floatVariables[subparameters[0] - 1],
                                                 x.GetY().floatVariables[subparameters[1] - 1]])
                        '''
                        self.points.append([x.GetY().floatVariables[subparameters[0] - 1],
                                       x.GetY().floatVariables[subparameters[1] - 1]])
                        self.values.append(x.GetZ())
                        self.pointsWithBestComb[0].append(x.GetY().floatVariables[subparameters[0] - 1])
                        self.pointsWithBestComb[1].append(x.GetY().floatVariables[subparameters[1] - 1])
                    else:
                        self.points.append(x.GetY().floatVariables[0])
                        self.values.append(x.GetZ())
                        self.pointsWithBestComb[0].append(x.GetY().floatVariables[0])
                        self.pointsWithBestComb[1].append(self.optimumVal - 5)

            if floatdim > 1:
                self.optimumPoint[0].append(self.optimum.floatVariables[subparameters[0] - 1])
                self.optimumPoint[1].append(self.optimum.floatVariables[subparameters[1] - 1])
            else:
                self.optimumPoint[0].append(self.optimum.floatVariables[0])
                self.optimumPoint[1].append(self.optimumVal - 5)

        elif mode == 'analysis':
            i = 0
            for item in searchDataSorted:
                i += 1
                if item.GetZ() > 1.7e+308:
                    continue
                self.points.append(item.GetY())
                self.values.append([item.GetZ(), i])
                str = '['
                for j in range(len(item.GetY().discreteVariables)):
                    str += item.GetY().discreteVariables[j] + ', '
                str = str[:-2]
                str += ']'
                self.combination.append([str, i])

        self.plotter = DisretePlotter(mode, pcount, floatdim, discreteValues, discreteName,
                                      subparameters, lb, rb, bestsvalues, self.numberOfParallelPoints)

    def PaintObjectiveFunc(self, numpoints):
        if self.calc == 'objective function':
            section = self.optimum
            self.plotter.PlotByGrid(self.CalculateFunc, section, numpoints)
        elif self.calc == 'interpolation':
            self.plotter.PlotInterpolation(self.points, self.values)

    def PaintPoints(self, mrks):
        self.plotter.PlotPoints(self.pointsWithBestComb, self.otherPoints, self.optimum, self.optimumPoint, mrks)

    def PaintAnalisys(self, mrks):
        self.plotter.PlotAnalisysSubplotsFigure(self.points, self.values,  self.combination, self.optimum, mrks)
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