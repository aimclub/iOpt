import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sko.GA import GA_TSP
from typing import Dict


class GA_TSP_2D(Problem):
    """
    Класс GA_TSP представляет возможность решения задачи коммивояжёра средствами генетического алгоритма.
      Найденное решение является оптимальным при использовании фиксированного значений количества итераций
      при варьировании значений вероятности мутации и размера популяции.
    """

    def __init__(self, cost_matrix: np.ndarray, num_iteration: int,
                 mutation_probability_bound: Dict[str, float],
                 population_size_bound: Dict[str, float]):
        """
        Конструктор класса GA_TSP_2D

        :param cost_matrix: Матрица расстояний
        :param num_iteration: Максимальное число итераций генетического алгоритма
        :param population_size_bound: Границы изменения размера популяции (low - нижняя граница, up - верхняя)
        :param mutation_probability_bound: Границы изменения вероятности мутации (low - нижняя граница, up - верхняя)
        """
        super(GA_TSP_2D, self).__init__()
        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.costMatrix = cost_matrix
        # Проверка корректности числа итераций метода
        if num_iteration <= 0:
            raise ValueError('The number of iterations cannot be zero or negative.')
        # Проверка валидности интервала вероятностей мутации
        if mutation_probability_bound['low'] > mutation_probability_bound['up'] or \
                mutation_probability_bound['low'] < 0.0 or \
                mutation_probability_bound['low'] > 1.0 or \
                mutation_probability_bound['up'] < 0.0 or \
                mutation_probability_bound['up'] > 1.0:
            raise ValueError('Invalid mutation probability interval')
        # Проверка валидности размера популяции
        if population_size_bound['low'] < 2 or population_size_bound['low'] > population_size_bound['up']:
            raise ValueError('Incorrect population sizes were established')

        self.numberOfIterations = num_iteration
        self.float_variable_names = np.array(["mutation probability", "population size"], dtype=str)
        self.lower_bound_of_float_variables = np.array([mutation_probability_bound['low'], population_size_bound['low']])
        self.upper_bound_of_float_variables = np.array([mutation_probability_bound['up'], population_size_bound['up'] / 2])
        self.n_dim = cost_matrix.shape[0]

    def calc_total_distance(self, routine):
        """
        Метод расчёта расстояния

        :param routine: массив вершин для подсчёта расстояния
        """
        num_points, = routine.shape
        return sum([self.costMatrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param function_value: объект хранения значения целевой функции в точке
        """
        mutation_prob, num_population = point.float_variables[0], int(point.float_variables[1])
        # Контроль чётности осыбей популяции
        if num_population % 2 != 0:
            num_population -= 1

        ga_tsp = GA_TSP(func=lambda x: self.calc_total_distance(x),
                        n_dim=self.n_dim, size_pop=num_population,
                        max_iter=int(self.numberOfIterations), prob_mut=mutation_prob)
        best_points, best_distance = ga_tsp.run()
        function_value.value = best_distance[0]
        #print(best_distance[0])
        return function_value
