from iOpt.method.search_data import SearchData
from iOpt.trial import Point, FunctionValue
from iOpt.solution import Solution

import matplotlib.pyplot as plt
import numpy as np
import os

class FunctionStaticPainter:
    def __init__(self, searchData: SearchData,
                 solution: Solution):
        self.searchData = searchData
        self.solution = solution

    def Paint(self, fileName):
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
        sv1d = StaticVisualization1D(points, bestTrialPoint, bestTrialValue, leftBound, rightBound, self.solution.problem.Calculate)
        sv1d.drawObjFunction(pointsCount=150)
        sv1d.drawPoints()
        if not os.path.isdir("output"):
            os.mkdir("output")
        plt.savefig("output\\" + fileName)
        
class StaticVisualization1D:
    def __init__(self, _points, _optimum, _optimumValue, _leftBound, _rightBound, _objFunc):
        self.points = _points
        self.optimum = _optimum
        self.optimumValue = _optimumValue
        self.leftBound = float(_leftBound[0])
        self.rightBound = float(_rightBound[0])
        self.objFunc = _objFunc

        plt.style.use('fivethirtyeight')
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim([self.leftBound, self.rightBound])
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
        plt.plot(x, z, linewidth = 2, color='blue')      

    def drawPoints(self):
        for point in self.points:
            self.ax.plot(point, self.optimumValue - 1, color='black', 
                        label='original', marker='o', markersize=1)     
        self.ax.plot(self.optimum, self.optimumValue - 1, color='red', 
                    label='original', marker='x', markersize=4)
    