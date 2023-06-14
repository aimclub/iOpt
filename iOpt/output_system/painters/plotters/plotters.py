import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from scipy import interpolate
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DisretePlotter:
    def __init__(self, mode, pcount, floatdim, parametersvals, parametersnames, id, subparameters, lb, rb):
        plt.style.use('fivethirtyeight')
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 6

        self.mode = mode
        self.subparameters = subparameters
        self.lb = lb
        self.rb = rb

        self.floatdim = floatdim

        self.comb = [list(x) for x in np.array(np.meshgrid(*parametersvals)).T.reshape(-1, len(parametersvals))]
        self.combcount = len(self.comb)

        if self.mode == 'analysis':
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.fig.suptitle('Analysis optimization method work', fontsize=10)
            self.axes = []

            self.count = 3
            self.pcount = pcount
            if self.count > pcount: self.count = pcount
            for i in range(self.count):
                self.axes.append(plt.subplot2grid((self.count, self.count + 1), (0, i), colspan=1, rowspan=1))
                plt.tight_layout()
                self.axes[i].set_xlabel('values of parameter ' + str(i + 1) + ' (p' + str(i + 1) + ') ' + parametersnames[i] + '')
                self.axes[i].set_ylabel('objective function values')
            self.axes[0].set_title('Scatter of objective function values for different parameters values', loc='left', fontsize=8)

            self.axes.append(plt.subplot2grid((self.count, self.count + 1), (0, self.count), colspan=2, rowspan=self.count))
            self.axes[self.count].set_title('Iteration characteristic', fontsize=8)
            self.axes[self.count].set_xlabel('objective function values')
            self.axes[self.count].set_ylabel('iteration')
            self.axes.append(plt.subplot2grid((self.count, self.count + 1), (1, 0), colspan=self.count, rowspan=self.count - 1))
            self.axes[self.count + 1].set_title(str(self.combcount)+' combinations of parameters used at different iterations', fontsize=8)
            self.axes[self.count + 1].set_xlabel('iteration')
            self.axes[self.count + 1].set_ylabel('discrete parameters values')

        if self.mode == 'bestcombination':
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            plt.tight_layout()

        self.name = ['' + str(i + 1) + ' ' + parametersnames[i] + '' for i in range(len(parametersnames))]

    def PlotPoints(self, discretePoints, id, values, allpoints, allvalues, optimum, Xs, calculate, clr='blue', mrkr='o', mrkrs=3):
        if self.mode == 'analysis':
            for j in range(self.count):
                self.axes[j].scatter([x.discreteVariables[0] for x in allpoints], allvalues, s=mrkrs ** 2, color='black')
                self.axes[j].set_xlim([self.axes[j].get_xlim()[0] - 1, self.axes[j].get_xlim()[1] + 1])
            y = []
            z = []
            j = 1
            for x in allvalues:
                y.append(x)
                z.append(j)
                j += 1
            self.axes[self.count].plot(y, z, color='black', linewidth=1, alpha=1)

            '''
            z.clear()
            y.clear()
            j = 1
            for x in allpoints:
                for i in range(len(x.discreteVariables)):
                    y.append('p' + str(i + 1) + '=' + x.discreteVariables[i])
                    z.append(j)
                j += 1
            self.axes[self.count + 1].scatter(z, y, s=mrkrs**2)
            '''
            z.clear()
            y.clear()
            j = 1
            for x in allpoints:
                str = '['
                for i in range(len(x.discreteVariables)):
                    str += x.discreteVariables[i] + ', '
                str = str[:-2]
                str += ']'
                y.append(str)
                z.append(j)

                j += 1
            sc = self.axes[self.count + 1].scatter(z, y,c=allvalues, cmap ='plasma',s=mrkrs ** 2)
            self.fig.colorbar(sc, orientation='vertical')
            plt.tight_layout()

            combstrs=[]
            for x in self.comb:
                str = '['
                for i in x:
                    str += i + ', '
                str = str[:-2]
                str += ']'
                combstrs.append(str)
            self.axes[self.count + 1].scatter([allvalues[0]]*len(self.comb), combstrs, alpha=0)

    def PlotByGrid(self, calculate, optimum, bestcombination, other, pointsCount=100, mrkrs=3):
        if self.mode == 'bestcombination':
            if self.floatdim > 1:
                # линии уровня
                x1 = np.linspace(self.lb[self.subparameters[0]- 1], self.rb[self.subparameters[0]- 1], pointsCount)
                x2 = np.linspace(self.lb[self.subparameters[1]- 1], self.rb[self.subparameters[1]- 1], pointsCount)
                xv, yv = np.meshgrid(x1, x2)
                z = []

                fv = optimum.floatVariables.copy()
                for i in range(pointsCount):
                    z_ = []
                    for j in range(pointsCount):
                        fv[self.subparameters[0] - 1] = xv[i, j]
                        fv[self.subparameters[1] - 1] = yv[i, j]
                        z_.append(calculate(fv, optimum.discreteVariables))
                    z.append(z_)

                xx=self.ax.contour(x1, x2, z, linewidths=1, levels=10,cmap='plasma')
                self.fig.colorbar(
                    ScalarMappable(norm=xx.norm, cmap=xx.cmap),
                )
                '''
                cb.ax.plot(0.5, mean, 'w.') # my data is between 0 and 1
                cb.ax.plot([0, 1], [rms]*2, 'w') # my data is between 0 and 1
                '''

                # точки испытаний
                self.ax.scatter(other[0], other[1], s=mrkrs ** 2, color='grey',
                                              label='points with another disrete parameters combinations')
                self.ax.scatter(bestcombination[0], bestcombination[1], s=mrkrs ** 2, color='blue',
                                              label='points with '+ str(optimum.discreteVariables))
                self.ax.scatter([optimum.floatVariables[self.subparameters[0] - 1]],
                                              [optimum.floatVariables[self.subparameters[1] - 1]],
                                              s=mrkrs ** 2, color='red', label='best trial point')
                self.ax.set_title('Lines layers of objective function in section of optimum point with best parameters combination', fontsize=10)
                self.ax.set_xlabel('x' + str(self.subparameters[0]))
                self.ax.set_ylabel('x' + str(self.subparameters[1]))
                plt.tight_layout()
                legend_obj = plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8, bbox_to_anchor=(1, 1))
                legend_obj.set_draggable(True)
            else:
                x = np.linspace(self.lb[0], self.rb[0], pointsCount)
                z = []
                fv = optimum.floatVariables.copy()
                for i in range(pointsCount):
                    fv[self.index] = x[i]
                    z.append(calculate(fv))
                self.ax.plot(x, z, linewidth=1, color='black', alpha=0.7)

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