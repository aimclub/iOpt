from iOpt.method.search_data import SearchData
from iOpt.trial import Point, FunctionValue
from iOpt.solution import Solution

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.neural_network import MLPRegressor
from scipy import interpolate


class FunctionStaticPainter:
    def __init__(self, searchData: SearchData,
                 solution: Solution):
        self.searchData = searchData
        self.solution = solution

    def Paint(self, fileName, pathForSaves, isPointsAtBottom, parameterInNDProblem, toPaintObjFunc):
        # формируем массив точек итераций для графика
        points = []
        values = []
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
        sv1d = StaticVisualization1D(isPointsAtBottom, parameterInNDProblem, points[1:len(points) - 1],
                                     values[1:len(values) - 1], bestTrialPoint, bestTrialValue, leftBound, rightBound,
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

        plt.show()

    def PaintApproximation(self, fileName, pathForSaves, isPointsAtBottom, parameterInNDProblem):
        # формируем массив точек итераций для графика
        points = []
        values = []
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
        sv1d = StaticVisualization1D(isPointsAtBottom, parameterInNDProblem, points[1:len(points) - 1],
                                     values[1:len(values) - 1], bestTrialPoint, bestTrialValue, leftBound, rightBound,
                                     self.solution.problem.Calculate)

        X_train = []
        y_train = []
        for item in self.searchData:
            X_train.append(item.GetY().floatVariables[parameterInNDProblem])
            y_train.append(item.GetZ())

        X_train = np.array(X_train[1:-1])
        y_train = y_train[1:-1]

        nn = MLPRegressor(activation='logistic',  # can be tanh, identity, logistic, relu
                          solver='lbfgs',  # can be lbfgs, sgd , adam
                          alpha=0.001,
                          hidden_layer_sizes=(50,),
                          max_iter=5000,
                          tol=10e-8,
                          random_state=None)

        nn.fit(X_train[:, np.newaxis], y_train)
        size = 100
        x_plot = np.linspace(leftBound[parameterInNDProblem], rightBound[parameterInNDProblem], size)
        z = nn.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, z, color='black', linewidth=1, alpha=0.7)

        sv1d.drawPoints()

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        plt.show()

    def PaintInterpolation(self, fileName, pathForSaves, isPointsAtBottom, parameterInNDProblem):
        # формируем массив точек итераций для графика
        points = []
        values = []
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
        sv1d = StaticVisualization1D(isPointsAtBottom, parameterInNDProblem, points[1:len(points) - 1],
                                     values[1:len(values) - 1], bestTrialPoint, bestTrialValue, leftBound, rightBound,
                                     self.solution.problem.Calculate)

        X_train = []
        y_train = []
        for item in self.searchData:
            X_train.append(item.GetY().floatVariables[parameterInNDProblem])
            y_train.append(item.GetZ())

        X_train = np.array(X_train[1:-1])
        y_train = np.array(y_train[1:-1])

        # f = interpolate.interp1d(X_train, y_train, kind=3, fill_value="extrapolate")
        f = interpolate.interp1d(X_train, y_train, kind=3)
        size = 100
        x_plot = np.linspace(min(X_train), max(X_train), size)
        plt.plot(x_plot, f(x_plot), color='black', linewidth=1, alpha=0.7)

        sv1d.drawPoints()

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        plt.show()


class StaticVisualization1D:
    def __init__(self, isPointsAtBottom, parameterInNDProblem, _points, _values, _optimum, _optimumValue, _leftBound,
                 _rightBound, _objFunc):
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

    def drawObjFunction(self, pointsCount):
        x = np.arange(self.leftBound, self.rightBound, (self.rightBound - self.leftBound) / pointsCount)
        z = []
        x_: Point
        copy = self.optimum.copy()
        for i in range(pointsCount):
            if self.parameterInNDProblem is not None:
                copy[self.parameterInNDProblem] = x[i]
                x_ = Point(copy, [])
            else:
                x_ = Point([x[i]], [])
            fv = FunctionValue()
            fv = self.objFunc(x_, fv)
            z.append(fv.value)
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.plot(x, z, linewidth=1, color='black', alpha=0.7)

    def drawPoints(self):
        for i in range(len(self.points)):
            point_ = self.points[i]
            if not self.isPointsAtBottom:
                value = self.values[i]
            else:
                value = self.optimumValue - 1
            if self.parameterInNDProblem is not None:
                point = point_[self.parameterInNDProblem]
            else:
                point = point_

            self.ax.plot(point, value, color='blue',
                         label='original', marker='o', markersize=2)

        if not self.isPointsAtBottom:
            value = self.optimumValue
        else:
            value - self.optimumValue - 1
        self.ax.plot(self.optimum[self.parameterInNDProblem], value, color='red',
                     label='original', marker='x', markersize=4)


class FunctionStaticNDPainter:
    def __init__(self, searchData: SearchData,
                 solution: Solution):
        self.searchData = searchData
        self.solution = solution

    def PaintLL(self, fileName, pathForSaves, params):
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
        sv1d = StaticVisualizationND("2d", points, bestTrialPoint, bestTrialValue, leftBound, rightBound,
                                     self.solution.problem.Calculate, first, second)

        pointsCount = 150
        xF = np.arange(sv1d.leftBoundF, sv1d.rightBoundF, (sv1d.rightBoundF - sv1d.leftBoundF) / pointsCount)
        xS = np.arange(sv1d.leftBoundS, sv1d.rightBoundS, (sv1d.rightBoundS - sv1d.leftBoundS) / pointsCount)
        copy = sv1d.optimum.copy()
        xv, yv = np.meshgrid(xF, xS)
        z = []

        for i in range(pointsCount):
            z_ = []
            for j in range(pointsCount):
                fv = FunctionValue()
                copy[sv1d.first] = xv[i, j]
                copy[sv1d.second] = yv[i, j]
                x_ = Point(copy, [])
                fv = FunctionValue()
                fv = sv1d.objFunc(x_, fv)
                z_.append(fv.value)
            z.append(z_)
        sv1d.ax.contour(xF, xS, z, linewidths=1, levels=25, cmap=plt.cm.viridis)

        for point in sv1d.points:
            sv1d.ax.plot(point[sv1d.first], point[sv1d.second], color='blue',
                         label='original', marker='o', markersize=2)
        sv1d.ax.plot(sv1d.optimum[sv1d.first], sv1d.optimum[sv1d.second], color='red',
                     label='original', marker='x', markersize=4)

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        plt.show()

    def PaintLLI(self, fileName, pathForSaves, params):
        first = params[0]
        second = params[1]

        # оптимум
        bestTrialPoint = self.solution.bestTrials[0].point.floatVariables
        bestTrialValue = self.solution.bestTrials[0].functionValues[0].value

        # границы
        leftBound = self.solution.problem.lowerBoundOfFloatVariables
        rightBound = self.solution.problem.upperBoundOfFloatVariables

        # передаём точки, оптимум, границы и указатель на функцию для построения целевой функции
        sv1d = StaticVisualizationND("2d", [], bestTrialPoint, bestTrialValue, leftBound, rightBound,
                                     self.solution.problem.Calculate, first, second)

        # формируем массив точек итераций для графика
        X_train = []
        X = []
        Y = []
        y_train = []
        for item in self.searchData:
            X.append(item.GetY().floatVariables[first])
            Y.append(item.GetY().floatVariables[second])
            X_train.append([item.GetY().floatVariables[first], item.GetY().floatVariables[second]])
            y_train.append(item.GetZ())

        X_train = X_train[1:-1]
        y_train = y_train[1:-1]

        X = X[1:-1]
        Y = Y[1:-1]
        # interp2d
        interp = interpolate.Rbf(X, Y, y_train)

        size = 100
        x_x = np.linspace(leftBound[first], rightBound[first], size)
        y_y = np.linspace(leftBound[second], rightBound[second], size)
        xx, yy = np.meshgrid(x_x, y_y)
        zz = interp(xx, yy)

        sv1d.ax.contour(x_x, y_y, zz, linewidths=1, cmap=plt.cm.viridis)

        for point in sv1d.points:
            sv1d.ax.plot(point[sv1d.first], point[sv1d.second], color='blue',
                         label='original', marker='o', markersize=2)

        sv1d.ax.plot(sv1d.optimum[sv1d.first], sv1d.optimum[sv1d.second], color='red',
                     label='original', marker='x', markersize=4)

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        plt.show()

    def PaintApproximation(self, fileName, pathForSaves, params):
        first = params[0]
        second = params[1]

        # оптимум
        bestTrialPoint = self.solution.bestTrials[0].point.floatVariables
        bestTrialValue = self.solution.bestTrials[0].functionValues[0].value

        # границы
        leftBound = self.solution.problem.lowerBoundOfFloatVariables
        rightBound = self.solution.problem.upperBoundOfFloatVariables

        # передаём точки, оптимум, границы и указатель на функцию для построения целевой функции
        sv1d = StaticVisualizationND("3d", [], bestTrialPoint, bestTrialValue, leftBound, rightBound,
                                     self.solution.problem.Calculate, first, second)

        # формируем массив точек итераций для графика
        X_train = []
        X = []
        Y = []
        y_train = []
        for item in self.searchData:
            X.append(item.GetY().floatVariables[first])
            Y.append(item.GetY().floatVariables[second])
            X_train.append([item.GetY().floatVariables[first], item.GetY().floatVariables[second]])
            y_train.append(item.GetZ())

        X_train = X_train[1:-1]
        y_train = y_train[1:-1]

        X = X[1:-1]
        Y = Y[1:-1]

        nn = MLPRegressor(activation='logistic',  # can be tanh, identity, logistic, relu
                          solver='lbfgs',  # can be lbfgs, sgd , adam
                          alpha=0.001,
                          hidden_layer_sizes=(40,),
                          max_iter=10000,
                          tol=10e-6,
                          random_state=10)

        nn.fit(X_train, y_train)
        size = 50
        x_x = np.linspace(leftBound[first], rightBound[first], size)
        y_y = np.linspace(leftBound[second], rightBound[second], size)
        xx, yy = np.meshgrid(x_x, y_y)

        # np.c - cлияние осей X и Y в точки
        # ravel - развернуть (к одномерному массиву)
        xy = np.c_[xx.ravel(), yy.ravel()]

        # Делаем предсказание значений
        z = nn.predict(xy)
        z = z.reshape(size, size)

        # полученная аппроксимация
        sv1d.ax.plot_surface(xx, yy, z, cmap=plt.cm.viridis, alpha=0.6)
        sv1d.ax.tick_params(axis='both', labelsize=8)
        sv1d.ax.scatter(X, Y, y_train, color='blue', label='original', marker='o', s=2, alpha=1.0)
        sv1d.ax.scatter([bestTrialPoint[first]], [bestTrialPoint[second]], bestTrialValue, s=4, color='red',
                        label='original', marker='x', alpha=1.0)

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        plt.show()

    def PaintInterpolation(self, fileName, pathForSaves, params):
        first = params[0]
        second = params[1]

        # оптимум
        bestTrialPoint = self.solution.bestTrials[0].point.floatVariables
        bestTrialValue = self.solution.bestTrials[0].functionValues[0].value

        # границы
        leftBound = self.solution.problem.lowerBoundOfFloatVariables
        rightBound = self.solution.problem.upperBoundOfFloatVariables

        # передаём точки, оптимум, границы и указатель на функцию для построения целевой функции
        sv1d = StaticVisualizationND("3d", [], bestTrialPoint, bestTrialValue, leftBound, rightBound,
                                     self.solution.problem.Calculate, first, second)

        # формируем массив точек итераций для графика
        X_train = []
        X = []
        Y = []
        y_train = []
        for item in self.searchData:
            X.append(item.GetY().floatVariables[first])
            Y.append(item.GetY().floatVariables[second])
            X_train.append([item.GetY().floatVariables[first], item.GetY().floatVariables[second]])
            y_train.append(item.GetZ())

        X_train = X_train[1:-1]
        y_train = y_train[1:-1]

        X = X[1:-1]
        Y = Y[1:-1]
        # interp2d
        interp = interpolate.Rbf(X, Y, y_train)

        size = 50
        x_x = np.linspace(leftBound[first], rightBound[first], size)
        y_y = np.linspace(leftBound[second], rightBound[second], size)
        xx, yy = np.meshgrid(x_x, y_y)
        zz = interp(xx, yy)

        # полученная аппроксимация
        sv1d.ax.plot_surface(xx, yy, zz, cmap=plt.cm.viridis, alpha=0.6)
        sv1d.ax.tick_params(axis='both', labelsize=8)
        sv1d.ax.scatter(X, Y, y_train, color='blue', label='original', marker='o', s=2, alpha=1.0)
        sv1d.ax.scatter([bestTrialPoint[first]], [bestTrialPoint[second]], bestTrialValue, s=4, color='red',
                        label='original', marker='x', alpha=1.0)

        if not os.path.isdir(pathForSaves):
            if pathForSaves == "":
                plt.savefig(fileName)
            else:
                os.mkdir(pathForSaves)
                plt.savefig(pathForSaves + "/" + fileName)
        else:
            plt.savefig(pathForSaves + "/" + fileName)

        plt.show()


class StaticVisualizationND:
    def __init__(self, mode, _points, _optimum, _optimumValue, _leftBound, _rightBound, _objFunc, _firstParameter,
                 _secondParameter):
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
        self.fig = plt.subplot()
        if mode == "3d":
            self.ax = plt.subplot(projection='3d')
        elif mode == "2d":
            self.ax = plt.subplot()
        plt.rcParams["figure.figsize"] = (8, 6)
        self.ax.set_xlim([self.leftBoundF, self.rightBoundF])
        self.ax.set_ylim([self.leftBoundS, self.rightBoundS])
        self.ax.tick_params(axis='both', labelsize=8)
