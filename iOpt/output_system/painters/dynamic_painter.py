from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.problem import Problem
from iOpt.solution import Solution

import matplotlib.pyplot as plt
import numpy as np
import os


class FunctionAnimationPainter:
    def __init__(self, problem: Problem, isPointsAtBottom):
        self.problem = problem
        leftBound = problem.lowerBoundOfFloatVariables
        rightBound = problem.upperBoundOfFloatVariables
        self.isPointsAtBottom = isPointsAtBottom
        self.av1d = AnimateVisualization1D([], 0.0, 0.0, leftBound, rightBound, self.problem.Calculate)

    def PaintObjectiveFunc(self):
        self.av1d.drawObjFunction(pointsCount=150)

    def PaintPoint(self, savedNewPoints: SearchDataItem):
        x_ = savedNewPoints[0].GetY().floatVariables
        fv = savedNewPoints[0].GetZ()
        self.av1d.drawPoint(x_, fv, self.isPointsAtBottom)

    def PaintOptimum(self, solution: Solution, fileName, pathForSaves):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.av1d.drawOptimum(bestTrialPoint, bestTrialValue, fileName, pathForSaves, self.isPointsAtBottom)


class AnimateVisualization1D:
    def __init__(self, _points, _optimum, _optimumValue, _leftBound, _rightBound, _objFunc):
        self.points = []
        self.values = []
        self.optimum = _optimum
        self.optimumValue = _optimumValue
        self.leftBound = float(_leftBound[0])
        self.rightBound = float(_rightBound[0])
        self.objFunc = _objFunc

        plt.ion()
        plt.style.use('fivethirtyeight')
        self.fig, self.ax = plt.subplots()
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.leftBound, self.rightBound)
        self.ax.tick_params(axis='both', labelsize=8)

    def drawObjFunction(self, pointsCount):
        x = np.arange(self.leftBound, self.rightBound, (self.rightBound - self.leftBound) / pointsCount)
        z = []
        for i in range(pointsCount):
            x_ = Point([x[i]], [])
            fv = FunctionValue()
            fv = self.objFunc(x_, fv)
            z.append(fv.value)
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.plot(x, z, linewidth=1, color='black', alpha=0.7)

    def drawPoint(self, point, value, isPointsAtBottom):
        # self.ax.plot(point, value, color='green', 
        #        label='original', marker='o', markersize=1)
        if isPointsAtBottom:
            value = -1
        self.ax.plot(point, value, color='blue',
                     label='original', marker='o', markersize=2)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def drawOptimum(self, point, value, fileName, pathForSaves, isPointsAtBottom):
        if isPointsAtBottom:
            value = -1
        self.ax.plot(point, value, color='red',
                     label='original', marker='x', markersize=4)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # отключить интерактивный режим по завершению анимации
        plt.ioff()

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        # нужно, чтобы график не закрывался после завершения анимации
        # plt.show()


class FunctionAnimationNDPainter:
    def __init__(self, problem: Problem, parametersPair):
        self.problem = problem
        leftBound = problem.lowerBoundOfFloatVariables
        rightBound = problem.upperBoundOfFloatVariables
        self.av1d = AnimateVisualizationND([], 0.0, 0.0, leftBound, rightBound, self.problem.Calculate, parametersPair)

    def PaintObjectiveFunc(self, solution: Solution):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        self.av1d.drawObjFunction(150, bestTrialPoint)

    def PaintPoint(self, savedNewPoints: SearchDataItem):
        x_ = savedNewPoints[0].GetY().floatVariables
        fv = savedNewPoints[0].GetZ()
        self.av1d.drawPoint(x_, fv)

    def PaintOptimum(self, solution: Solution, fileName, pathForSaves):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.av1d.drawOptimum(bestTrialPoint, bestTrialValue, fileName, pathForSaves)


class AnimateVisualizationND:
    def __init__(self, _points, _optimum, _optimumValue, _leftBound, _rightBound, _objFunc, parameters):
        self.points = []
        self.values = []

        self.first = parameters[0]
        self.second = parameters[1]

        self.leftBoundF = float(_leftBound[self.first])
        self.rightBoundF = float(_rightBound[self.first])
        self.leftBoundS = float(_leftBound[self.second])
        self.rightBoundS = float(_rightBound[self.second])
        self.objFunc = _objFunc

        plt.ion()
        plt.style.use('fivethirtyeight')
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.leftBoundF, self.rightBoundF)
        self.ax.set_ylim(self.leftBoundS, self.rightBoundS)
        self.ax.tick_params(axis='both', labelsize=8)

    def drawObjFunction(self, pointsCount, optimum):
        xF = np.arange(self.leftBoundF, self.rightBoundF, (self.rightBoundF - self.leftBoundF) / pointsCount)
        xS = np.arange(self.leftBoundS, self.rightBoundS, (self.rightBoundS - self.leftBoundS) / pointsCount)
        copy = optimum.copy()
        xv, yv = np.meshgrid(xF, xS)
        z = []

        for i in range(pointsCount):
            z_ = []
            for j in range(pointsCount):
                fv = FunctionValue()
                copy[self.first] = xv[i, j]
                copy[self.second] = yv[i, j]
                x_ = Point(copy, [])
                fv = FunctionValue()
                fv = self.objFunc(x_, fv)
                z_.append(fv.value)
            z.append(z_)
        self.ax.contour(xF, xS, z, linewidths=1, levels=25, cmap=plt.cm.viridis)

    def drawPoint(self, point, value):
        self.ax.plot(point[self.first], point[self.second], color='blue',
                     label='original', marker='o', markersize=2)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def drawOptimum(self, point, value, fileName, pathForSaves):
        self.ax.plot(point[self.first], point[self.second], color='red',
                     label='original', marker='x', markersize=4)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # отключить интерактивный режим по завершению анимации
        plt.ioff()

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(os.path.curdir + "/" + fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        # нужно, чтобы график не закрывался после завершения анимации
        # plt.show()
