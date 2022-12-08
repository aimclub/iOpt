import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from sko.GA import GA_TSP
from typing import Dict

class GA_TSP_2D(Problem):

    def __init__(self, cost_matrix: np.ndarray, num_iteration: int,
                 mutation_probability_bound: Dict[str, float],
                 population_size_bound: Dict[str, float]):
        self.dimension = 2
        self.numberOfFloatVariables = 2
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        self.costMatrix = cost_matrix
        # Проверка корректности числа итераций метода
        if num_iteration <= 0:
            raise ValueError('The number of iterations cannot be zero or negative.')
        #Проверка валидности интервала вероятностей мутации
        if mutation_probability_bound['low'] > mutation_probability_bound['up'] or mutation_probability_bound['low'] < 0.0 or mutation_probability_bound['low'] > 1.0 or mutation_probability_bound['up'] < 0.0 or mutation_probability_bound['up'] > 1.0:
            raise ValueError('Invalid mutation probability interval')
        # Проверка валидности размера популяции
        if population_size_bound['low'] < 2 or population_size_bound['low'] > population_size_bound['up']:
            raise ValueError('Incorrect population sizes were established')

        self.numberOfIterations = num_iteration
        self.floatVariableNames = np.array(["mutation probability", "population size"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([mutation_probability_bound['low'], population_size_bound['low']])
        self.upperBoundOfFloatVariables = np.array([mutation_probability_bound['up'], population_size_bound['up'] / 2])
        self.n_dim = cost_matrix.shape[0]



    def calc_total_distance(self, routine):
        num_points, = routine.shape
        return sum([self.costMatrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        mutation_prob, num_population = point.floatVariables[0], int(point.floatVariables[1])
        #Контроль чётности осыбей популяции
        if num_population % 2 != 0:
            num_population -= 1

        ga_tsp = GA_TSP(func=self.calc_total_distance,
                        n_dim=self.n_dim, size_pop=num_population,
                        max_iter=int(self.numberOfIterations), prob_mut=mutation_prob)
        best_points, best_distance = ga_tsp.run()
        functionValue.value = best_distance[0]
        print(best_distance[0])
        return functionValue
