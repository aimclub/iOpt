import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict
from sklearn.model_selection import StratifiedKFold

class SVC_2D_Transformators_State(Problem):
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
        super(SVC_2D_Transformators_State, self).__init__()
        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.float_variable_names = np.array(["Regularization parameter", "Kernel coefficient"], dtype=str)
        self.lower_bound_of_float_variables = np.array([regularization_bound['low'], kernel_coefficient_bound['low']],
                                                   dtype=np.double)
        self.upper_bound_of_float_variables = np.array([regularization_bound['up'], kernel_coefficient_bound['up']],
                                                   dtype=np.double)

        self.cv = StratifiedKFold(shuffle=True, random_state=42)



    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param function_value: объект хранения значения целевой функции в точке
        """
        cs, gammas = point.float_variables[0], point.float_variables[1]
        clf = SVC(C=10 ** cs, gamma=10 ** gammas)
        function_value.value = -cross_val_score(clf, self.x, self.y, cv=self.cv, scoring='f1_macro').mean()
        return function_value