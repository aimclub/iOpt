from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


# import time

class FunctionConsoleFullOutput:
    def __init__(self, problem: Problem, parameters: SolverParameters):
        self.problem = problem
        self.parameters = parameters
        self.__outputer = ConsoleOutputer()
        self.iterNum = 1

    def printInitInfo(self):
        self.__outputer.printInit(
            self.parameters.eps,
            self.parameters.r,
            self.parameters.epsR,
            self.parameters.itersLimit,

            self.problem.numberOfFloatVariables,
            self.problem.numberOfObjectives,
            self.problem.numberOfConstraints,
            self.problem.lowerBoundOfFloatVariables,
            self.problem.upperBoundOfFloatVariables
        )

    def printIterPointInfo(self, savedNewPoints: SearchDataItem):
        point = savedNewPoints[0].GetY().floatVariables
        value = savedNewPoints[0].GetZ()

        self.__outputer.printIter(
            point,
            value,
            self.iterNum
        )

        self.iterNum += 1

    def printBestPointInfo(self, solution, iters):
        if self.iterNum % iters != 0:
            pass
        else:
            bestTrialPoint = solution.bestTrials[0].point.floatVariables
            bestTrialValue = solution.bestTrials[0].functionValues[0].value
            self.__outputer.printBest(
                solution.numberOfGlobalTrials,
                solution.numberOfLocalTrials,
                solution.solutionAccuracy,
                bestTrialPoint,
                bestTrialValue,
                self.iterNum
            )
        self.iterNum += 1

    def printFinalResult(self, solution: Solution, status: bool):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.__outputer.printResult(
            status,
            solution.numberOfGlobalTrials,
            solution.numberOfLocalTrials,
            solution.solvingTime,
            solution.solutionAccuracy,
            bestTrialPoint,
            bestTrialValue
        )


class ConsoleOutputer:
    def printInit(self, eps, r, epsR, itersLimit, floatdim, numberOfObjectives, numberOfConstraints,
                  lowerBoundOfFloatVariables, upperBoundOfFloatVariables):
        dim = floatdim
        print()
        print("-" * (30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Task Description", width=30 + 20 * dim))
        print("-" * (30 + 20 * dim + 2))
        print("|{:>29} {:<{width}}|".format("dimension: ", floatdim, width=20 * dim))
        tempstr = "["
        for i in range(floatdim):
            tempstr += "["
            tempstr += str(lowerBoundOfFloatVariables[i])
            tempstr += ", "
            tempstr += str(upperBoundOfFloatVariables[i])
            tempstr += "], "
        tempstr = tempstr[:-2]
        tempstr += "]"
        print("|{:>29} {:<{width}}|".format("bounds: ", tempstr, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("objective-function count: ", numberOfObjectives, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("constraint-function count: ", numberOfConstraints, width=20 * dim))
        print("-" * (30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Method Parameters", width=30 + 20 * dim))
        print("-" * (30 + 20 * dim + 2))
        print("|{:>29} {:<{width}}|".format("eps: ", eps, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("r: ", r, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("epsR: ", epsR, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("itersLimit: ", itersLimit, width=20 * dim))
        print("-" * (30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Iterations", width=30 + 20 * dim))
        print("-" * (30 + 20 * dim + 2))
        print("|{:^{width}}|".format("", width=30 + 20 * dim))

    def printIter(self, point, value, iter):
        dim = len(point)

        print("|", end=' ')
        # print("\033[A|", end=' ')
        print("{:>5}:".format(iter), end=' ')
        print("{:>19.8f}".format(value), end='   ')
        print("{:<{width}}|".format(str(point), width=20 * dim))

    def printResult(self, solved, numberOfGlobalTrials, numberOfLocalTrials, solvingTime, solutionAccuracy,
                    bestTrialPoint, bestTrialValue):
        dim = len(bestTrialPoint)
        print("-" * (30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Result", width=30 + 20 * dim))
        print("-" * (30 + 20 * dim + 2))
        # print("|{:>29} {:<{width}}|".format("is solved: ", str(solved), width=20*dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", numberOfGlobalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", numberOfLocalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("solving time: ", solvingTime, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("solution point: ", str(bestTrialPoint), width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("solution value: ", bestTrialValue, width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("accuracy: ", solutionAccuracy, width=20 * dim))
        print("-" * (30 + 20 * dim + 2))
        pass

    def printBest(self, numberOfGlobalTrials, numberOfLocalTrials, solutionAccuracy,
                  bestTrialPoint, bestTrialValue, iter):
        dim = len(bestTrialPoint)
        print("|{:>29} {:<{width}}|".format("current iteration # ", iter, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", numberOfGlobalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", numberOfLocalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("current best point: ", str(bestTrialPoint), width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("current best value: ", bestTrialValue, width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("currant accuracy: ", solutionAccuracy, width=20 * dim))
        print("." * (30 + 20 * dim + 2))
