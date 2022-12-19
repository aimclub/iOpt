import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sko.GA import GA_TSP
from typing import Dict


class GA_TSP_Vary_Mutation(Problem):
    """
    Класс GA_TSP_Vary_Mutatiom представляет возможность решения задачи коммивояжёра средствами генетического алгоритма.
      Найденное решение является оптимальным при использовании фиксированных значений количества итераций и размера
      популяции на обозначенном отрезке варьирования вероятности мутации.
    """

    def __init__(self, cost_matrix: np.ndarray, num_iteration: int, population_size: int,
                 mutation_probability_bound: Dict[str, float]):
        """
        Конструктор класса GA_TSP_Vary_Mutation

        :param cost_matrix: Матрица расстояний
        :param num_iteration: Максимальное число итераций генетического алгоритма
        :param population_size: Размер популяции
        :param mutation_probability_bound: Границы изменения вероятности мутации (low - нижняя граница, up - верхняя)
        """
        super(GA_TSP_Vary_Mutation, self).__init__()
        self.dimension = 1
        self.numberOfFloatVariables = 1
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        self.costMatrix = cost_matrix
        if num_iteration <= 0:
            raise ValueError('The number of iterations cannot be zero or negative.')
        if population_size <= 0:
            raise ValueError('Population size cannot be negative or zero')
        self.populationSize = population_size
        self.numberOfIterations = num_iteration
        self.floatVariableNames = np.array(["Mutation probability"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([mutation_probability_bound['low']], dtype=np.double)
        self.upperBoundOfFloatVariables = np.array([mutation_probability_bound['up']], dtype=np.double)
        self.n_dim = cost_matrix.shape[0]

    def calc_total_distance(self, routine):
        """
        Метод расчёта расстояния

        :param routine: массив вершин для подсчёта расстояния
        """
        num_points, = routine.shape
        return sum([self.costMatrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """
        Метод расчёта значения целевой функции в точке

        :param point: Точка испытания
        :param functionValue: объект хранения значения целевой функции в точке
        """
        mutation_prob = point.floatVariables[0]
        ga_tsp = GA_TSP(func=self.calc_total_distance,
                        n_dim=self.n_dim, size_pop=self.populationSize,
                        max_iter=self.numberOfIterations, prob_mut=mutation_prob)
        best_points, best_distance = ga_tsp.run()
        functionValue.value = best_distance[0]
        #        print(best_distance[0])
        return functionValue
