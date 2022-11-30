from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.problem import Problem
from iOpt.solution import Solution

import matplotlib.pyplot as plt
import numpy as np
import os

class FunctionAnimationPainter:
    def __init__(self, problem : Problem):
        self.problem = problem
        leftBound = problem.lowerBoundOfFloatVariables
        rightBound = problem.upperBoundOfFloatVariables
        self.av1d = AnimateVisualization1D([], 0.0, 0.0, leftBound, rightBound, self.problem.Calculate)
    
    def PaintObjectiveFunc(self):
        self.av1d.drawObjFunction(pointsCount=150)

    def PaintPoint(self, savedNewPoints : SearchDataItem):
        x_ = savedNewPoints[0].GetY().floatVariables
        fv = savedNewPoints[0].GetZ()
        self.av1d.drawPoint(x_, fv)

    def PaintOptimum(self, solution : Solution, fileName):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.av1d.drawOptimum(bestTrialPoint, bestTrialValue, fileName)
        
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
        self.ax.tick_params(axis = 'both', labelsize = 8)
    
    def drawObjFunction(self, pointsCount):
        x = np.arange(self.leftBound, self.rightBound, (self.rightBound - self.leftBound) / pointsCount)
        z = []
        for i in range(pointsCount):
            x_ = Point([x[i]], [])
            fv = FunctionValue()
            fv = self.objFunc(x_, fv)
            z.append(fv.value)
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.plot(x, z, linewidth=1, color='blue')

    def drawPoint(self, point, value):
        #self.ax.plot(point, value, color='green', 
        #        label='original', marker='o', markersize=1)
        self.ax.plot(point, -1, color='black', 
                label='original', marker='o', markersize=1)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    def drawOptimum(self, point, value, fileName):
        self.ax.plot(point, -1, color='red', 
                label='original', marker='x', markersize=4)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # отключить интерактивный режим по завершению анимации
        plt.ioff()
        # нужно, чтобы график не закрывался после завершения анимации
        # plt.show()

        if not os.path.isdir("output"):
            os.mkdir("output")
        plt.savefig("output\\" + fileName)