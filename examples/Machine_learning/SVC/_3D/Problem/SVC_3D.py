import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, List

class SVC_3D(Problem):
    """
    Класс SVC_3D представляет возможность поиска оптимального набора гиперпараметров алгоритма
      C-Support Vector Classification.
      Найденные параметры являются оптимальными при варьировании параматра регуляризации
      (Regularization parameter С) значения коэфицента ядра (gamma) и типа ядра (kernel)
    """

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray,
                 regularization_bound: Dict[str, float],
                 kernel_coefficient_bound: Dict[str, float],
                 kernel_type: Dict[str, List[str]]
                 ):
        """
        Конструктор класса SVC_3D

        :param x_dataset: входные данные обучающе выборки метода SVC
        :param y_dataset: выходные данные обучающе выборки метода SVC
        :param kernel_coefficient_bound: Значение параметра регуляризации
        :param regularization_bound: Границы изменения значений коэфицента ядра (low - нижняя граница, up - верхняя)
        :param kernel_type: Тип ядра, используемый в алгоритме SVC
        """
        super(SVC_3D, self).__init__()
        self.dimension = 3
        self.numberOfFloatVariables = 2
        self.numberOfDiscreteVariables = 1
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
        self.discreteVariableNames.append('kernel')
        self.discreteVariableValues.append(kernel_type['kernel'])


    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param functionValue: объект хранения значения целевой функции в точке
        """
        cs, gammas = point.floatVariables[0], point.floatVariables[1]
        kernel_type = point.discreteVariables[0]
        clf = SVC(C=10 ** cs, gamma=10 ** gammas, kernel=kernel_type)
        functionValue.value = -cross_val_score(clf, self.x, self.y, scoring='f1').mean()
        return functionValue
