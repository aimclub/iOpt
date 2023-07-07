from typing import List

from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters

class ConsoleOutputer:
    def __init__(self, problem: Problem, parameters: SolverParameters):
        self.problem = problem
        self.parameters = parameters
        self.__functions = OutputFunctions()
        self.iterNum = 1
        self.ndv = self.problem.number_of_discrete_variables

    def PrintInitInfo(self):
        self.__functions.printInit(
            self.parameters.eps,
            self.parameters.r,
            self.parameters.eps_r,
            self.parameters.iters_limit,
            self.problem.number_of_float_variables,
            self.problem.number_of_objectives,
            self.problem.number_of_constraints,
            self.problem.lower_bound_of_float_variables,
            self.problem.upper_bound_of_float_variables,
            self.problem.number_of_discrete_variables,
            self.parameters.number_of_parallel_points
        )

    def PrintIterPointInfo(self, savedNewPoints: List[SearchDataItem]):
        if self.parameters.number_of_parallel_points > 1:
            isFirst = True
        else:
            isFirst = False

        for i in range(len(savedNewPoints)):
            point = savedNewPoints[i].GetY().float_variables
            dpoint = savedNewPoints[i].GetY().discrete_variables
            value = savedNewPoints[i].GetZ()

            self.__functions.printIter(
                point,
                dpoint,
                value,
                self.iterNum, self.ndv,
                isFirst
            )
            isFirst = False

            self.iterNum += 1

    def PrintBestPointInfo(self, solution, iters):
        if self.iterNum % iters != 0:
            pass
        else:
            bestTrialPoint = solution.best_trials[0].point.float_variables
            bestTrialDPoint = solution.best_trials[0].point.discrete_variables
            bestTrialValue = solution.best_trials[0].function_values[0].value
            self.__functions.printBest(
                solution.number_of_global_trials,
                solution.number_of_local_trials,
                solution.solution_accuracy,
                bestTrialPoint,
                bestTrialDPoint,
                bestTrialValue,
                self.iterNum, self.ndv
            )
        self.iterNum += 1

    def PrintFinalResultInfo(self, solution: Solution, status: bool):
        bestTrialPoint = solution.best_trials[0].point.float_variables
        bestTrialDPoint = solution.best_trials[0].point.discrete_variables
        bestTrialValue = solution.best_trials[0].function_values[0].value
        self.__functions.printResult(
            status,
            solution.number_of_global_trials,
            solution.number_of_local_trials,
            solution.solving_time,
            solution.solution_accuracy,
            bestTrialPoint,
            bestTrialDPoint,
            bestTrialValue, self.ndv
        )


class OutputFunctions:

    def printInit(self, eps, r, eps_r, iters_limit, floatdim, number_of_objectives, number_of_constraints,
                  lower_bound_of_float_variables, upper_bound_of_float_variables, number_of_discrete_variables,
                  number_of_parallel_points):
        dim = floatdim + number_of_discrete_variables
        size_max_one_output = 15
        print()
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Task Description", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:>29} {:<{width}}|".format("dimension: ", floatdim, width=size_max_one_output * dim))
        tempstr = "["
        for i in range(floatdim):
            tempstr += "["
            tempstr += str(lower_bound_of_float_variables[i])
            tempstr += ", "
            tempstr += str(upper_bound_of_float_variables[i])
            tempstr += "], "
        tempstr = tempstr[:-2]
        tempstr += "]"
        print("|{:>29} {:<{width}}|".format("bounds: ", tempstr, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("objective-function count: ", number_of_objectives, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("constraint-function count: ", number_of_constraints, width=size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Method Parameters", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:>29} {:<{width}}|".format("eps: ", eps, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("r: ", r, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("eps_r: ", eps_r, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("iters_limit: ", iters_limit, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("number_of_parallel_points: ", number_of_parallel_points, width=size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Iterations", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("", width=30 + size_max_one_output * dim))

    def printIter(self, point, dpoint, value, iter, ndv, flag):
        size_max_one_output = 15
        dim1 = len(point)
        if dpoint:
            dim2 = len(dpoint)
        else:
            dim2 = 0
        print("|", end=' ')
        # print("\033[A|", end=' ')
        if flag:
            print("*{:>4}:".format(iter), end=' ')
        else:
            print("{:>5}:".format(iter), end=' ')
        print("{:>19.8f}".format(value), end='   ')
        if ndv > 0:
            print("{:<{width}}|".format(str(point) + " with " + str(dpoint), width = size_max_one_output * (dim1 + dim2)))
        else:
            print("{:<{width}}|".format(str(point), width=size_max_one_output * dim1))

    def printResult(self, solved, number_of_global_trials, number_of_local_trials, solving_time, solution_accuracy,
                    bestTrialPoint, bestTrialDPoint, bestTrialValue, ndv):
        size_max_one_output = 15
        dim = len(bestTrialPoint) + len(bestTrialDPoint)
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Result", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        # print("|{:>29} {:<{width}}|".format("is solved: ", str(solved), width=20*dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", number_of_global_trials, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", number_of_local_trials, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("solving time: ", solving_time, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("solution point: ", str(bestTrialPoint), width=size_max_one_output * dim))
        if ndv > 0:
            print("|{:>29} {:<{width}}|".format("best disrete combination: ", str(bestTrialDPoint), width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("solution value: ", bestTrialValue, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("accuracy: ", solution_accuracy, width=size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))

    def printBest(self, number_of_global_trials, number_of_local_trials, solution_accuracy,
                  bestTrialPoint, bestTrialDPoint, bestTrialValue, iter, ndv):
        size_max_one_output = 15
        dim = len(bestTrialPoint) + len(bestTrialDPoint)
        print("|{:>29} {:<{width}}|".format("current iteration # ", iter, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", number_of_global_trials, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", number_of_local_trials, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("current best point: ", str(bestTrialPoint), width=size_max_one_output * dim))
        if ndv > 0:
            print("|{:>29} {:<{width}}|".format("with discrete combination: ", str(bestTrialDPoint), width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("current best value: ", bestTrialValue, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("currant accuracy: ", solution_accuracy, width=size_max_one_output * dim))
        print("." * (30 + size_max_one_output * dim + 2))
