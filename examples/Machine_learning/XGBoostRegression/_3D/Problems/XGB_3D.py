import numpy as np
import xgboost as xgb
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class XGB_3D(Problem):
    """
    Класс XGB_3D представляет возможность поиска оптимального набора гиперпараметров алгоритма XGBoost regression
      Найденные параметры являются оптимальными при варьировании параматра скорости обучения
      (learning_rate), коэфицента определения минимального снижения потерь (gamma) и типа модели (booster)
    """

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray,
                 learning_rate_bound: Dict[str, float],
                 gamma_bound: Dict[str, float],
                 booster_type: Dict[str, List[str]]
                 ):
        """
        Конструктор класса SVC_3D

        :param x_dataset: входные данные обучающе выборки метода SVC
        :param y_dataset: выходные данные обучающе выборки метода SVC
        :param learning_rate_bound: Границы изменения значений параметра скорости обучения
        :param gamma_bound: Границы изменения значений параметра gамма, определяющего минимальное снижение потерь
        :param booster_type: Тип модели, для каждой итерации алгоритма XGBoost
        """
        super(XGB_3D, self).__init__()
        self.dimension = 3
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.float_variable_names = np.array(["learning_rate", "gamma"], dtype=str)
        self.lower_bound_of_float_variables = np.array([learning_rate_bound['low'], gamma_bound['low']],
                                                   dtype=np.double)
        self.upper_bound_of_float_variables = np.array([learning_rate_bound['up'], gamma_bound['up']],
                                                   dtype=np.double)
        self.discrete_variable_names.append('booster')
        self.discrete_variable_values.append(booster_type['booster'])


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param function_value: объект хранения значения целевой функции в точке
        """
        learning_rate, gamma = point.float_variables[0], point.float_variables[1]
        booster = point.discrete_variables[0]
        regr = xgb.XGBRegressor(learning_rate=learning_rate, gamma=gamma, booster=booster)
        function_value.value = -cross_val_score(regr, self.x, self.y, scoring='r2', cv=5).mean()
        return function_value
