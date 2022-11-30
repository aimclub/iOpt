from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.trial import Point, FunctionValue
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.method.method import Method
from iOpt.solver_parametrs import SolverParameters

import matplotlib.pyplot as plt
import numpy as np
import time
import os

class Listener:
    def BeforeMethodStart(self, searchData: SearchData):
        pass

    def OnEndIteration(self, searchData: SearchData):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass

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

class FunctionConsoleFullOutput:
    def __init__(self, problem : Problem, parameters : SolverParameters):
        self.problem = problem
        self.parameters = parameters
        self.__outputer = ConsolePrint()
        self.iterNum = 1

    def printInitInfo(self):
        self.__outputer.printInit(
            self.parameters.eps,
            self.parameters.r,
            self.parameters.epsR,
            self.parameters.itersLimit,

            self.problem.numberOfFloatVariables,
            self.problem.numberOfObjectives,
            self.problem.numberOfConstraints,
            self.problem.lowerBoundOfFloatVariables,
            self.problem.upperBoundOfFloatVariables
        )

    def printIterInfo(self, savedNewPoints : SearchDataItem, mode):
        point = savedNewPoints[0].GetY().floatVariables
        value = savedNewPoints[0].GetZ()

        self.__outputer.printIter(
            point,
            value,
            self.iterNum,
            mode
        )
        self.iterNum += 1

    def printFinalResult(self, solution : Solution, status: bool):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.__outputer.printResult(
            status,
            solution.numberOfGlobalTrials,
            solution.numberOfLocalTrials,
            solution.solvingTime,
            solution.solutionAccuracy,
            bestTrialPoint,
            bestTrialValue
        )

class ConsolePrint:
    def printInit(self, eps, r, epsR, itersLimit, floatdim, numberOfObjectives, numberOfConstraints,
        lowerBoundOfFloatVariables, upperBoundOfFloatVariables):
        dim = floatdim
        print()
        print("-"*(30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Task Discription", width=30+20*dim))
        print("-"*(30 + 20 * dim + 2))
        print("|{:>29} {:<{width}}|".format("dimension: ", floatdim, width=20*dim))
        tempstr = "["
        for i in range (floatdim):
            tempstr += "["
            tempstr += str(lowerBoundOfFloatVariables[i])
            tempstr += ", "
            tempstr += str(upperBoundOfFloatVariables[i])
            tempstr += "], "
        tempstr = tempstr[:-2]
        tempstr += "]"
        print("|{:>29} {:<{width}}|".format("bounds: ", tempstr, width=20*dim))
        print("|{:>29} {:<{width}}|".format("objective-function count: ", numberOfObjectives, width=20*dim))
        print("|{:>29} {:<{width}}|".format("constraint-function count: ", numberOfConstraints, width=20*dim))
        print("-"*(30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Method Parameters", width=30+20*dim))
        print("-"*(30 + 20 * dim + 2))
        print("|{:>29} {:<{width}}|".format("eps: ", eps, width=20*dim))
        print("|{:>29} {:<{width}}|".format("r: ", r, width=20*dim))
        print("|{:>29} {:<{width}}|".format("epsR: ", epsR, width=20*dim))
        print("|{:>29} {:<{width}}|".format("itersLimit: ", itersLimit, width=20*dim))
        print("-"*(30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Iterations", width=30+20*dim))
        print("-"*(30 + 20 * dim + 2))
        print()
        pass

    def printIter(self, point, value, iter, mode):
        time.sleep(0.1)
        dim = len(point)
        if mode == 1:
            print("|", end=' ')
        elif mode == 2:
            print("\033[A|", end=' ')
        print("{:>5}:".format(iter), end=' ')
        print("{:>19.8f}".format(value), end ='   ')
        print("{:<{width}}|".format(str(point), width=20*dim))

    def printResult(self, solved, numberOfGlobalTrials, numberOfLocalTrials, solvingTime, solutionAccuracy,
        bestTrialPoint, bestTrialValue):
        dim = len(bestTrialPoint)
        print("-"*(30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Result",width=30+20*dim))
        print("-"*(30 + 20 * dim + 2))
        print("|{:>29} {:<{width}}|".format("is solved: ", str(solved), width=20*dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", numberOfGlobalTrials, width=20*dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", numberOfLocalTrials, width=20*dim))
        print("|{:>29} {:<{width}}|".format("solving time: ", solvingTime, width=20*dim))
        print("|{:>29} {:<{width}}|".format("solution point: ", str(bestTrialPoint), width=20*dim))
        print("|{:>29} {:<{width}.8f}|".format("solution value: ", bestTrialValue, width=20*dim))
        print("|{:>29} {:<{width}.8f}|".format("accuracy: ", solutionAccuracy, width=20*dim))
        print("-"*(30 + 20 * dim + 2))
        pass

class ConsoleFullOutputListener(Listener):
    def __init__(self, mode):
        self.__fcfo : FunctionConsoleFullOutput = None
        self.mode = mode

    def BeforeMethodStart(self, method : Method):
        self.__fcfo = FunctionConsoleFullOutput(method.task.problem, method.parameters)
        self.__fcfo.printInitInfo()
        pass

    def OnEndIteration(self, savedNewPoints : SearchDataItem):
        self.__fcfo.printIterInfo(savedNewPoints, self.mode)
        pass

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status: bool):
        self.__fcfo.printFinalResult(solution, status)
        pass

class PaintListener(Listener):
    def __init__(self, fileName):
        self.fileName = fileName

    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution, status : bool):
        fp = FunctionStaticPainter(searchData, solution)
        fp.Paint(self.fileName)

class AnimationPaintListener(Listener):
    def __init__(self, fileName):
        self.__fp : FunctionAnimationPainter = None
        self.fileName = fileName

    def BeforeMethodStart(self, method : Method):
        self.__fp = FunctionAnimationPainter(method.task.problem)
        self.__fp.PaintObjectiveFunc()

    def OnEndIteration(self, savedNewPoints : SearchDataItem):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status : bool):
        self.__fp.PaintOptimum(solution, self.fileName)

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
