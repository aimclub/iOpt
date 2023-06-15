import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from scipy import interpolate
from matplotlib.cm import ScalarMappable

class DisretePlotter:
    def __init__(self, mode, pcount, floatdim, parametersvals, parametersnames, subparameters, lb, rb, bestsvalues):
        plt.style.use('fivethirtyeight')
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 6

        self.subparameters = subparameters
        self.lb = lb
        self.rb = rb
        self.floatdim = floatdim
        self.bestsvalues=bestsvalues

        self.discreteParamsCombinations = [list(x) for x in np.array(np.meshgrid(*parametersvals)).T.reshape(-1,
             len(parametersvals))]
        self.combcount = len(self.discreteParamsCombinations)

        if mode == 'analysis':
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.fig.suptitle('Analysis optimization method work', fontsize=10)
            self.axes = []

            self.count = 4
            self.pcount = pcount
            if self.count > pcount: self.count = pcount
            for i in range(self.count):
                self.axes.append(plt.subplot2grid((4, 4), (0, i), colspan=1, rowspan=1))
                plt.tight_layout()
                self.axes[i].set_xlabel('values of parameter ' + str(i + 1)
                                        + parametersnames[i] + '')
                self.axes[i].set_ylabel('objective function values')
            self.axes[0].set_title('Scatter of objective function values for different parameters values',
                                   loc='left', fontsize=8)

            self.axes.append(plt.subplot2grid((4, 4), (1,2), colspan=2, rowspan=2))
            self.axes[self.count].set_title('Iteration characteristic', fontsize=8)
            self.axes[self.count].set_xlabel('iteration')
            self.axes[self.count].set_ylabel('objective function values')

            self.axes.append(plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=3))
            self.axes[self.count + 1].set_title(str(self.combcount)+
                                                ' combinations of parameters used at different iterations', fontsize=8)
            self.axes[self.count + 1].set_xlabel('iteration')
            self.axes[self.count + 1].set_ylabel('discrete parameters values')

            self.axes.append(plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=1))
            self.axes[self.count + 2].set_title('Current best value update', fontsize=8)
            self.axes[self.count + 2].set_xlabel('iteration')
            self.axes[self.count + 2].set_ylabel('best minimum value')

        elif mode == 'bestcombination':
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            plt.tight_layout()

        self.name = ['' + str(i + 1) + ' ' + parametersnames[i] + '' for i in range(len(parametersnames))]

    def PlotAnalisysSubplotsFigure(self, allpoints, allvalues, combinations, mrkrs=3):
            for j in range(self.count):
                self.axes[j].scatter([x.discreteVariables[0] for x in allpoints],
                                     [item[0] for item in allvalues],
                                     s=mrkrs ** 2, color='black')
                self.axes[j].set_xlim([self.axes[j].get_xlim()[0] - 1, self.axes[j].get_xlim()[1] + 1])

            self.axes[self.count].plot([item[1] for item in allvalues],
                                       [item[0] for item in allvalues],
                                       color='black', linewidth=1, alpha=1)
            sc = self.axes[self.count + 1].scatter([item[1] for item in combinations],
                                                   [item[0] for item in combinations],
                                                   c=[item[0] for item in allvalues],
                                                   cmap ='plasma',s=mrkrs ** 2)

            self.fig.colorbar(sc, orientation='vertical')
            plt.tight_layout()

            iters = list(range(1, len(self.bestsvalues) + 1))
            self.axes[self.count + 2].plot(iters, self.bestsvalues, color='black', linewidth=1, alpha=1)

            combinations = []
            for x in self.discreteParamsCombinations:
                str = '['
                for i in x:
                    str += i + ', '
                str = str[:-2]
                str += ']'
                combinations.append(str)
            self.axes[self.count + 1].scatter([allvalues[0][0]] * self.combcount, combinations, alpha=0)

    def PlotPoints(self, best, other, optimum, optimumPoint, mrkrs):
            self.ax.scatter(other[0], other[1], s=mrkrs ** 2, color='grey',
                            label='points with another discrete parameters combinations')
            self.ax.scatter(best[0], best[1], s=mrkrs ** 2, color='blue',
                            label='points with ' + str(optimum.discreteVariables))
            self.ax.scatter(optimumPoint[0],
                            optimumPoint[1],
                            s=mrkrs ** 2, color='red', label='best trial point')
            plt.tight_layout()
            legend_obj = plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8, bbox_to_anchor=(1, 1))
            legend_obj.set_draggable(True)

    def PlotByGrid(self, calculate, section, pointsCount):
            if self.floatdim > 1:
                i = self.subparameters[0]
                j = self.subparameters[1]

                xi = np.linspace(self.lb[i - 1], self.rb[i - 1], pointsCount)
                xj = np.linspace(self.lb[j - 1], self.rb[j - 1], pointsCount)
                xv, yv = np.meshgrid(xi, xj)

                z = []
                fv = section.floatVariables.copy()
                for k in range(pointsCount):
                    z_ = []
                    for t in range(pointsCount):
                        fv[i - 1] = xv[k, t]
                        fv[j - 1] = yv[k, t]
                        z_.append(calculate(fv, section.discreteVariables))
                    z.append(z_)

                self.ax.set_title(
                    'Lines layers of objective function in section of optimum point with best parameters combination',
                    fontsize=10)
                self.ax.set_xlabel('x' + str(i))
                self.ax.set_ylabel('x' + str(j))

                xx = self.ax.contour(xi, xj, z, linewidths=1, levels=10, cmap='plasma')

                self.fig.colorbar(ScalarMappable(norm=xx.norm, cmap=xx.cmap))

            else:
                x = np.linspace(self.lb[0], self.rb[0], pointsCount)
                z = []
                fv = section.floatVariables.copy()
                for k in range(pointsCount):
                    fv[0] = x[k]
                    z.append(calculate(fv, section.discreteVariables))

                self.ax.set_title(
                    'Objective function with best discrete parameters combination',
                    fontsize=10)
                self.ax.set_xlabel('trial point')
                self.ax.set_ylabel('objective function value')

                self.ax.plot(x, z, linewidth=1, color='black', alpha=0.7)

    def PlotInterpolation(self, points, values, pointsCount=100):
            if self.floatdim > 1:
                i = self.subparameters[0] - 1
                j = self.subparameters[1] - 1
                interp = interpolate.Rbf(np.array(points)[:, 0], np.array(points)[:, 1], values)
                x1 = np.linspace(self.lb[i], self.rb[i], pointsCount)
                x2 = np.linspace(self.lb[j], self.rb[j], pointsCount)
                x1, x2 = np.meshgrid(x1, x2)
                z = interp(x1, x2)
                self.ax.contour(x1, x2, z, levels=10, linewidths=1, cmap='plasma')
            else:
                f = interpolate.interp1d(np.array(points), np.array(values), kind=3)
                x_plot = np.linspace(min(np.array(points)), max(np.array(points)), pointsCount)
                plt.plot(x_plot, f(x_plot), color='black', linewidth=1, alpha=0.7)

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