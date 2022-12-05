import numpy as np
import numpy.typing as npt
from sko.GA import GA_TSP

from iOpt.problem import Problem
from iOpt.trial import FunctionValue, Point


class GA_TSP_1D(Problem):

    def __init__(
            self,
            dimension: int,
            costmatrix: npt.NDArray[np.double],
            numiteration: int, populationsize: int
    ):
        self.dimension = 1
        self.numberOfFloatVariables = 1
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        self.costMatrix = costmatrix
        if numiteration <= 0:
            raise ValueError('The number of iterations cannot be zero or negative.')
        if populationsize <= 0:
            raise ValueError('Population size cannot be negative or zero')
        self.populationSize = populationsize
        self.numberOfIterations = numiteration
        self.floatVariableNames = ["mutation probability"]
        self.lowerBoundOfFloatVariables = [0.0]
        self.upperBoundOfFloatVariables = [1.0]

    def calc_total_distance(self, routine: npt.NDArray[np.int32]) -> float:
        num_points, = routine.shape
        return sum([self.costMatrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        ga_tsp = GA_TSP(func=self.calc_total_distance,
                        n_dim=280, size_pop=self.populationSize,
                        max_iter=self.numberOfIterations,
                        prob_mut=point.floatVariables[0])
        best_points, best_distance = ga_tsp.run()
        functionValue.value = best_distance[0]
        return functionValue
