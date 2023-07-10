import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict


class SVC_Fixed_Kernel(Problem):
    """
    Класс SVC_Fixed_Kernel представляет возможность поиска оптимального набора гиперпараметров алгоритма
      C-Support Vector Classification.
      Найденные параметры являются оптимальными при фиксированном значении коэфицента ядра метода (gamma)
      при варьировании параматра регуляризации (Regularization parameter С)
    """

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, kernel_coefficient_value: float,
                 regularization_bound: Dict[str, float]):
        """
        Конструктор класса SVC_Fixed_Kernel

        :param x_dataset: входные данные обучающе выборки метода SVC
        :param y_dataset: выходные данные обучающе выборки метода SVC
        :param kernel_coefficient_value: Значение коэфицента ядра
        :param regularization_bound: Границы изменения параметра регуляризации (low - нижняя граница, up - верхняя)
        """
        super(SVC_Fixed_Kernel, self).__init__()
        self.dimension = 1
        self.number_of_float_variables = 1
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.kernelCoefficient = kernel_coefficient_value
        self.float_variable_names = np.array(["Regularization parameter"], dtype=str)
        self.lower_bound_of_float_variables = np.array([regularization_bound['low']], dtype=np.double)
        self.upper_bound_of_float_variables = np.array([regularization_bound['up']], dtype=np.double)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param function_value: объект хранения значения целевой функции в точке
        """
        cs = point.float_variables[0]
        clf = SVC(C=10 ** cs, gamma=10 ** self.kernelCoefficient)
        clf.fit(self.x, self.y)
        function_value.value = -cross_val_score(clf, self.x, self.y,
                                                scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()
        return function_value
