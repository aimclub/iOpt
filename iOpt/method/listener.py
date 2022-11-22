from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.trial import Trial, Point, FunctionValue
from iOpt.problem import Problem
from iOpt.solution import Solution

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import numpy as np
import time

#интерфейс в методе
class Listener:
    def BeforeMethodStart(self, searchData: SearchData,):
        pass

    def OnEndIteration(self, searchData: SearchData):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass

#реализацию вынести за метод!
class FunctionStaticPainter:
    def __init__(self, searchData: SearchData,
                 solution: Solution):
        self.searchData = searchData
        self.solution = solution

    def Paint(self):
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
        plt.savefig("output\methodWorks.pdf")

class FunctionAnimationPainter:
    def __init__(self, problem : Problem):
        self.problem = problem
        leftBound = problem.lowerBoundOfFloatVariables
        rightBound = problem.upperBoundOfFloatVariables
        self.av1d = AnimateVisualization1D([], 0.0, 0.0, leftBound, rightBound, self.problem.Calculate)
    
    def PaintObjectiveFunc(self):
        self.av1d.drawObjFunction(pointsCount=150)

    def PaintPoint(self, savedNewPoints):
        x_ = Point(savedNewPoints, [])
        fv = FunctionValue()
        fv = self.problem.Calculate(x_, fv)
        self.av1d.drawPoint(savedNewPoints[0], fv.value)

    def PaintOptimum(self, solution : Solution):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.av1d.drawOptimum(bestTrialPoint, bestTrialValue)

# пример слушателя
class PaintListener(Listener):
    # нарисовать все точки испытаний
    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution):
        fp = FunctionStaticPainter(searchData, solution)
        fp.Paint()

class AnimationPaintListener(Listener):
    __fp : FunctionAnimationPainter = None

    def BeforeMethodStart(self, problem : Problem):
        self.__fp = FunctionAnimationPainter(problem)
        self.__fp.PaintObjectiveFunc()

    def OnEndIteration(self, savedNewPoints):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData : SearchData, solution: Solution):
        self.__fp.PaintOptimum(solution)

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

        #time.sleep(1)
        

    def drawOptimum(self, point, value):
        self.ax.plot(point, -1, color='red', 
                label='original', marker='x', markersize=4)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
         # Отключить интерактивный режим по завершению анимации
        plt.ioff()
        # Нужно, чтобы график не закрывался после завершения анимации
        #plt.show()
        plt.savefig("output\methodWorks.pdf")
        pass