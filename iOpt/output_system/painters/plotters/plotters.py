import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from scipy import interpolate

class Plotter:
    """
    Базовый класс вызовов функций стандартного плоттера matplotlib.pyplot.
    """
    def PlotByGrid(self):
        pass
    def PlotApproximation(self):
        pass
    def PlotInterpolation(self):
        pass
    def PlotPoints(self):
        pass

class Plotter2D(Plotter):
    def __init__(self, parameterInNDProblem, leftBound, rightBound):
        plt.style.use('fivethirtyeight')
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.rcParams["figure.figsize"] = (8, 6)

        self.index = parameterInNDProblem
        self.leftBound = leftBound
        self.rightBound = rightBound

        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.tick_params(axis='both', labelsize=8)
        self.ax.set_xlim([self.leftBound, self.rightBound])

    def PlotByGrid(self, calculate, section, pointsCount=100):
        x = np.arange(self.leftBound, self.rightBound, (self.rightBound - self.leftBound) / pointsCount)
        z = []
        for i in range(pointsCount):
            section[self.index] = x[i]
            z.append(calculate(section))
        plt.plot(x, z, linewidth=1, color='black', alpha=0.7)

    def PlotApproximation(self, points, values, pointsCount=100):
        nn = MLPRegressor(activation='logistic',  # can be tanh, identity, logistic, relu
                          solver='lbfgs',  # can be lbfgs, sgd , adam
                          alpha=0.001,
                          hidden_layer_sizes=(50,),
                          max_iter=5000,
                          tol=10e-8,
                          random_state=None)
        nn.fit(np.array(points)[:, np.newaxis], np.array(values))
        x_plot = np.linspace(self.leftBound, self.rightBound, pointsCount)
        z = nn.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, z, color='black', linewidth=1, alpha=0.7)

    def PlotInterpolation(self, points, values, pointsCount=100):
        f = interpolate.interp1d(np.array(points), np.array(values), kind=3)
        x_plot = np.linspace(min(np.array(points)), max(np.array(points)), pointsCount)
        plt.plot(x_plot, f(x_plot), color='black', linewidth=1, alpha=0.7)

    def PlotPoints(self, points, values, clr='blue', mrkr='o', mrkrs=4):
        self.ax.scatter(points, values, color=clr, marker=mrkr, s=mrkrs)

class Plotter3D(Plotter):
    def __init__(self, parametersInNDProblem, leftBounds, rightBounds, objFunc, plotterType):
        plt.style.use('fivethirtyeight')
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.rcParams["figure.figsize"] = (8, 6)

        self.indexes = parametersInNDProblem
        self.leftBounds = leftBounds
        self.rightBounds = rightBounds
        self.objFunc = objFunc

        self.plotterType = plotterType

        self.fig = plt.subplot()
        if self.plotterType == 'surface':
            self.ax = plt.subplot(projection='3d')
        elif self.plotterType == 'lines layers':
            self.ax = plt.subplot()
        self.ax.set_xlim([self.leftBounds[0], self.rightBounds[0]])
        self.ax.set_ylim([self.leftBounds[1], self.rightBounds[1]])
        self.ax.tick_params(axis='both', labelsize=8)

    def PlotByGrid(self, calculate, section, pointsCount=100):
        x1 = np.arange(self.leftBounds[0], self.rightBounds[0], (self.rightBounds[0] - self.leftBounds[0]) / pointsCount)
        x2 = np.arange(self.leftBounds[1], self.rightBounds[1], (self.rightBounds[1] - self.leftBounds[1]) / pointsCount)
        xv, yv = np.meshgrid(x1, x2)
        z = []

        for i in range(pointsCount):
            z_ = []
            for j in range(pointsCount):
                section[self.indexes[0]] = xv[i, j]
                section[self.indexes[1]] = yv[i, j]
                z_.append(calculate(section))
            z.append(z_)

        self.ax.contour(x1, x2, z, linewidths=1, levels=25, cmap=plt.cm.viridis)

    def PlotApproximation(self, points, values, pointsCount=100):
        nn = MLPRegressor(activation='logistic',  # can be tanh, identity, logistic, relu
                          solver='lbfgs',  # can be lbfgs, sgd , adam
                          alpha=0.001,
                          hidden_layer_sizes=(40,),
                          max_iter=10000,
                          tol=10e-6,
                          random_state=10)

        nn.fit(points, values)
        x1 = np.linspace(self.leftBounds[0], self.rightBounds[0], pointsCount)
        x2 = np.linspace(self.leftBounds[1], self.rightBounds[1], pointsCount)
        x1, x2 = np.meshgrid(x1, x2)

        # np.c - cлияние осей X и Y в точки
        # ravel - развернуть (к одномерному массиву)
        xy = np.c_[x1.ravel(), x2.ravel()]

        # делаем предсказание значений
        z = nn.predict(xy)
        z = z.reshape(pointsCount, pointsCount)

        # полученная аппроксимация
        self.ax.plot_surface(x1, x2, z, cmap=plt.cm.viridis, alpha=0.6)

    def PlotInterpolation(self, points, values, pointsCount=100):
        if self.plotterType == 'lines layers':
            interp = interpolate.Rbf(np.array(points)[:, 0], np.array(points)[:, 1], values)
            x1 = np.linspace(self.leftBounds[0], self.rightBounds[0], pointsCount)
            x2 = np.linspace(self.leftBounds[1], self.rightBounds[1], pointsCount)
            x1, x2 = np.meshgrid(x1, x2)
            z = interp(x1, x2)
            self.ax.contour(x1, x2, z, linewidths=1, cmap=plt.cm.viridis)
        elif self.plotterType == 'surface':
            interp = interpolate.Rbf(np.array(points)[:, 0], np.array(points)[:, 1], values)
            x1 = np.linspace(self.leftBounds[0], self.rightBounds[0], pointsCount)
            x2 = np.linspace(self.leftBounds[1], self.rightBounds[1], pointsCount)
            x1, x2 = np.meshgrid(x1, x2)
            z = interp(x1, x2)
            self.ax.plot_surface(x1, x2, z, cmap=plt.cm.viridis, alpha=0.6)

    def PlotPoints(self, points, values, clr='blue', mrkr='o', mrkrs=3):
        if self.plotterType == 'lines layers':
            self.ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], color=clr, marker=mrkr, s=mrkrs)
        elif self.plotterType == 'surface':
            self.ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], values, s=mrkrs, color=clr, marker=mrkr, alpha=1.0)

class AnimatePlotter2D(Plotter2D):
    def __init__(self, parametersInNDProblem, leftBounds, rightBounds, objFunc=None, plotterType='lines layers'):
        plt.ion()
        plt.style.use('fivethirtyeight')
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.rcParams["figure.figsize"] = (8, 6)

        self.index = parametersInNDProblem
        self.leftBound = 0
        self.rightBound = 0
        self.objFunc = objFunc

        self.plotterType = plotterType

        self.fig, self.ax = plt.subplots()
        self.ax.tick_params(axis='both', labelsize=8)
        self.ax.set_autoscaley_on(True)

    def SetBounds(self, leftBound, rightBound):
        self.leftBound = leftBound
        self.rightBound = rightBound

        self.ax.set_xlim(self.leftBound, self.rightBound)

    def PlotPoints(self, points, values, clr='blue', mrkr='o', mrkrs=4):
        self.ax.plot(points[0], values[0], color=clr, marker=mrkr, markersize=mrkrs)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class AnimatePlotter3D(Plotter3D):
    def __init__(self, parametersInNDProblem, objFunc=None, plotterType='lines layers'):
        plt.ion()
        plt.style.use('fivethirtyeight')
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.rcParams["figure.figsize"] = (8, 6)

        self.indexes = parametersInNDProblem
        self.leftBounds = [0, 0]
        self.rightBounds = [1, 1]
        self.objFunc = None

        self.fig, self.ax = plt.subplots()

        self.ax.tick_params(axis='both', labelsize=8)
        self.ax.set_autoscaley_on(True)

    def SetBounds(self, leftBounds, rightBounds):
        self.leftBounds = leftBounds
        self.rightBounds = rightBounds

        self.ax.set_xlim(self.leftBounds[0], self.rightBounds[0])
        self.ax.set_ylim(self.leftBounds[1], self.rightBounds[1])

    def PlotPoints(self, points, values, clr='blue', mrkr='o', mrkrs=4):
        self.ax.plot(points[0], points[1], color=clr, marker=mrkr, markersize=mrkrs)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()