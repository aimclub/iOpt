import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from sko.GA import GA_TSP
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict


class SVC_Fixed_Regularization(Problem):
    """
    Класс SVC_Fixed_Regularization представляет возможность поиска оптимального набора гиперпараметров алгоритма
      C-Support Vector Classification.
      Найденные параметры являются оптимальными при фиксированном значении параметра регуляризации (С) при
      варьировании значения коэфицента ядра (gamma)
    """

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, regularization_value: float,
                 kernel_coefficient_bound: Dict[str, float]):
        """
        Конструктор класса SVC_Fixed_Regularization

        :param x_dataset: входные данные обучающе выборки метода SVC
        :param y_dataset: выходные данные обучающе выборки метода SVC
        :param regularization_value: Значение параметра регуляризации
        :param kernel_coefficient_bound: Границы изменения значений коэфицента ядра (low - нижняя граница, up - верхняя)
        """
        super(SVC_Fixed_Regularization, self).__init__()
        self.dimension = 1
        self.numberOfFloatVariables = 1
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.regularizationValue = regularization_value
        self.floatVariableNames = np.array(["Kernel coefficient"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([kernel_coefficient_bound['low']], dtype=np.double)
        self.upperBoundOfFloatVariables = np.array([kernel_coefficient_bound['up']], dtype=np.double)
        s = 0

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param functionValue: объект хранения значения целевой функции в точке
        """
        kernel_coefficient = point.floatVariables[0]
        clf = SVC(C=10 ** self.regularizationValue, gamma=10 ** kernel_coefficient)
        clf.fit(self.x, self.y)
        functionValue.value = -cross_val_score(clf, self.x, self.y,
                                               scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()
        return functionValue
