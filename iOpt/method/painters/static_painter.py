import os
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np

from iOpt.method.search_data import SearchData
from iOpt.solution import Solution
from iOpt.trial import FunctionValue, Point


class FunctionStaticPainter:
    def __init__(self, searchData: SearchData,
                 solution: Solution):
        self.searchData = searchData
        self.solution = solution

    def Paint(self, fileName: str, pathForSaves: str, isPointsAtBottom: bool,
              parameterInNDProblem: int, toPaintObjFunc: bool) -> None:
        # формируем массив точек итераций для графика
        points: List[List[float]] = []
        values: List[float] = []
        for item in self.searchData:
            points.append(item.GetY().floatVariables)
            if not isPointsAtBottom:
                values.append(item.GetZ())

        # оптимум
        bestTrialPoint = self.solution.bestTrials[0].point.floatVariables
        bestTrialValue = self.solution.bestTrials[0].functionValues[0].value

        # границы
        leftBound = self.solution.problem.lowerBoundOfFloatVariables
        rightBound = self.solution.problem.upperBoundOfFloatVariables

        # передаём точки, оптимум, границы и указатель на функцию для построения целевой функции
        sv1d = StaticVisualization1D(isPointsAtBottom,
                                     parameterInNDProblem,
                                     points[1:len(points) - 1],
                                     values[1:len(values) - 1],
                                     bestTrialPoint,
                                     bestTrialValue,
                                     leftBound,
                                     rightBound,
                                     self.solution.problem.Calculate)
        if toPaintObjFunc:
            sv1d.drawObjFunction(pointsCount=150)
        sv1d.drawPoints()

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)


class StaticVisualization1D:
    def __init__(self, isPointsAtBottom: bool, parameterInNDProblem: int,
                 _points: List[List[float]], _values: List[float],
                 _optimum: List[float], _optimumValue: float,
                 _leftBound: List[float], _rightBound: List[float],
                 _objFunc: Callable[[Point, FunctionValue], FunctionValue]) -> None:
        self.points = _points
        self.values = _values
        self.isPointsAtBottom = isPointsAtBottom
        self.optimum = _optimum
        self.optimumValue = _optimumValue
        self.leftBound = float(_leftBound[parameterInNDProblem])
        self.rightBound = float(_rightBound[parameterInNDProblem])
        self.objFunc = _objFunc
        self.parameterInNDProblem = parameterInNDProblem

        plt.style.use('fivethirtyeight')
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim([self.leftBound, self.rightBound])
        self.ax.tick_params(axis='both', labelsize=8)

    def drawObjFunction(self, pointsCount: int) -> None:
        x = np.arange(self.leftBound, self.rightBound, (self.rightBound - self.leftBound) / pointsCount)
        z = []
        x_: Point
        copy = self.optimum.copy()
        for i in range(pointsCount):
            copy[self.parameterInNDProblem] = x[i]
            x_ = Point(copy, [])
            fv = FunctionValue()
            fv = self.objFunc(x_, fv)
            z.append(fv.value)
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.plot(x, z, linewidth=1, color='black', alpha=0.7)

    def drawPoints(self) -> None:
        for i in range(len(self.points)):
            point_ = self.points[i]
            if not self.isPointsAtBottom:
                value = self.values[i]
            else:
                value = self.optimumValue - 1
            point = point_[self.parameterInNDProblem]
            self.ax.plot(point, value, color='blue',
                         label='original', marker='o', markersize=1)

        if not self.isPointsAtBottom:
            value = self.optimumValue
        else:
            value - self.optimumValue - 1
        self.ax.plot(self.optimum[self.parameterInNDProblem], value, color='red',
                     label='original', marker='x', markersize=4)


class FunctionStaticNDPainter:
    def __init__(self, searchData: SearchData, solution: Solution) -> None:
        self.searchData = searchData
        self.solution = solution

    def Paint(self, fileName: str, pathForSaves: str, params: List[int], toPaintObjFunc: bool) -> None:
        first = params[0]
        second = params[1]
        # формируем массив точек итераций для графика
        points = []
        for item in self.searchData:
            points.append(item.GetY().floatVariables)

        # оптимум
        bestTrialPoint = self.solution.bestTrials[0].point.floatVariables
        bestTrialValue = self.solution.bestTrials[0].functionValues[0].value

        # границы
        leftBound = self.solution.problem.lowerBoundOfFloatVariables
        rightBound = self.solution.problem.upperBoundOfFloatVariables

        # передаём точки, оптимум, границы и указатель на функцию для построения целевой функции
        sv1d = StaticVisualizationND(
            points,
            bestTrialPoint,
            bestTrialValue,
            leftBound,
            rightBound,
            self.solution.problem.Calculate,
            first,
            second)
        if toPaintObjFunc:
            sv1d.drawObjFunction(pointsCount=150)
        sv1d.drawPoints()
        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)


class StaticVisualizationND:
    def __init__(self, _points: List[List[float]], _optimum: List[float],
                 _optimumValue: float, _leftBound: List[float],
                 _rightBound: List[float],
                 _objFunc: Callable[[Point, FunctionValue], FunctionValue],
                 _firstParameter: int, _secondParameter: int) -> None:
        self.points = _points
        self.optimum = _optimum
        self.optimumValue = _optimumValue
        self.leftBoundF = float(_leftBound[_firstParameter])
        self.rightBoundF = float(_rightBound[_firstParameter])
        self.leftBoundS = float(_leftBound[_secondParameter])
        self.rightBoundS = float(_rightBound[_secondParameter])
        self.objFunc = _objFunc
        self.first = _firstParameter
        self.second = _secondParameter

        plt.style.use('fivethirtyeight')
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim([self.leftBoundF, self.rightBoundF])
        self.ax.set_ylim([self.leftBoundS, self.rightBoundS])
        self.ax.tick_params(axis='both', labelsize=8)

    def drawObjFunction(self, pointsCount: int) -> None:
        xF = np.arange(self.leftBoundF, self.rightBoundF, (self.rightBoundF - self.leftBoundF) / pointsCount)
        xS = np.arange(self.leftBoundS, self.rightBoundS, (self.rightBoundS - self.leftBoundS) / pointsCount)
        copy = self.optimum.copy()
        xv, yv = np.meshgrid(xF, xS, indexing='xy')
        z = []

        for i in range(pointsCount):
            z_ = []
            for j in range(pointsCount):
                fv = FunctionValue()
                copy[self.first] = xv[j, i]
                copy[self.second] = yv[j, i]
                x_ = Point(copy, [])
                fv = FunctionValue()
                fv = self.objFunc(x_, fv)
                z_.append(fv.value)
            z.append(z_)
        self.ax.contour(xF, xS, z, linewidths=1, cmap=plt.cm.viridis)

    def drawPoints(self) -> None:
        for point in self.points:
            self.ax.plot(point[self.first], point[self.second], color='blue',
                         label='original', marker='o', markersize=1)
        self.ax.plot(self.optimum[self.first], self.optimum[self.second], color='red',
                     label='original', marker='x', markersize=4)
