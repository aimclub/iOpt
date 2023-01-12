import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SVC_2D(Problem):
    """
    Класс SVC_2D представляет возможность поиска оптимального набора гиперпараметров алгоритма
      C-Support Vector Classification.
      Найденные параметры являются оптимальными при варьировании параматра регуляризации
      (Regularization parameter С) значения коэфицента ядра (gamma)
    """

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray,
                 regularization_bound: Dict[str, float],
                 kernel_coefficient_bound: Dict[str, float]):
        """
        Конструктор класса SVC_2D

        :param x_dataset: входные данные обучающе выборки метода SVC
        :param y_dataset: выходные данные обучающе выборки метода SVC
        :param kernel_coefficient_bound: Значение параметра регуляризации
        :param regularization_bound: Границы изменения значений коэфицента ядра (low - нижняя граница, up - верхняя)
        """
        super(SVC_2D, self).__init__()
        self.dimension = 2
        self.numberOfFloatVariables = 2
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.floatVariableNames = np.array(["Regularization parameter", "Kernel coefficient"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([regularization_bound['low'], kernel_coefficient_bound['low']],
                                                   dtype=np.double)
        self.upperBoundOfFloatVariables = np.array([regularization_bound['up'], kernel_coefficient_bound['up']],
                                                   dtype=np.double)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param functionValue: объект хранения значения целевой функции в точке
        """
        cs, gammas = point.floatVariables[0], point.floatVariables[1]
        clf = Pipeline([('scaler', StandardScaler()), ('model', SVC(C=10 ** cs, gamma=10 ** gammas))])
        functionValue.value = -cross_val_score(clf, self.x, self.y, scoring='f1').mean()
        return functionValue
