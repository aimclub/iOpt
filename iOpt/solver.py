from typing import List
import numpy as np
from iOpt.method.listener import Listener
from iOpt.trial import Point
from iOpt.problem import Problem
from iOpt.solution import Solution

class SolverParameters:
    def __init__(self,
                 eps: np.double = 0.01,
                 r: np.double = 2.0,
                 itersLimit: int = 20000,
                 evolventDensity: int = 10,
                 epsR: np.double = 0.001,
                 refineSolution: bool = False,
                 startPoint: Point = []
                ):
        """
        :param eps:method tolerance. Less value -- better search precision, less probability of early stop.
        :param r: reliability parameter. Higher value of r -- slower convergence, higher chance to cache the global minima.
        :param itersLimit: max number of iterations.
        :param evolventDensity:density of evolvent. By default density is 2^-12 on hybercube [0,1]^N,
               which means that maximum search accuracyis 2^-12.
               If search hypercube is large the density can be increased accordingly to achieve better accuracy.
        :param epsR: parameter which prevents method from paying too much attention to constraints. Greater values of
               this parameter speed up convergence, but global minima can be lost.
        :param refineSolution: if true, the final solution will be refined with the HookJeves method.
        """
        self.eps = eps
        self.r = r
        self.itersLimit = itersLimit
        self.evolventDensity = evolventDensity
        self.epsR = epsR
        self.refineSolution = refineSolution
        self.startPoint = startPoint



class Solver:
    __listeners: List[Listener] = []

    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters = SolverParameters()
                ):
        """
        :param problem: Optimization problem
        :param parameters: Parameters for solving the problem
        """
        self.problem = problem
        self.parameters = parameters

    def Solve(self) -> Solution:
        """
        Retrieve a solution with check of the stop conditions
        :return: Solution for the optimization problem
        """

    def DoGlobalIteration(self, number: int = 1):
        """
        :param number: The number of iterations of the global search
        """

    def DoLocalRefinement(self, number: int = 1):
        """
        :param number: The number of iterations of the local search
        """

    def GetResults(self) -> Solution:
        """
        :return: Return current solution for the optimization problem
        """

    def SaveProgress(self, fileName: str):
        """
        :return:
        """

    def LoadProgress(self, fileName: str):
        """
        :return:
        """

    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        self.__listeners.append(listener)
