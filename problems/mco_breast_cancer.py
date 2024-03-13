import numpy as np

from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC


class mco_breast_cancer(Problem):

    def __init__(self, X, y, X_train, y_train):
        """
        Конструктор класса breast_cancer problem.
        """

        super(mco_breast_cancer, self).__init__()

        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train

        self.name = "mco_test1"
        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)

        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.array([1, -7], dtype=np.double)
        self.upper_bound_of_float_variables = np.array([6, -3], dtype=np.double)

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        result: np.double = 0
        x = point.float_variables

        svc_c = 10 ** x[0]
        gamma = 10 ** x[1]

        classifier_obj = SVC(C=svc_c, gamma=gamma)
        classifier_obj.fit(self.X_train, self.y_train)

        if function_value.functionID == 0:  # OBJECTIV 1
            result = - cross_val_score(classifier_obj, self.X, self.y, n_jobs=4, scoring='precision').mean()
        elif function_value.functionID == 1:  # OBJECTIV 2
            result = - cross_val_score(classifier_obj, self.X, self.y, n_jobs=4, scoring='recall').mean()

        function_value.value = result
        return function_value
