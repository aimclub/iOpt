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
        self.ndv = self.problem.numberOfDisreteVariables

    def PrintInitInfo(self):
        self.__functions.printInit(
            self.parameters.eps,
            self.parameters.r,
            self.parameters.epsR,
            self.parameters.itersLimit,
            self.problem.numberOfFloatVariables,
            self.problem.numberOfObjectives,
            self.problem.numberOfConstraints,
            self.problem.lowerBoundOfFloatVariables,
            self.problem.upperBoundOfFloatVariables,
            self.problem.numberOfDisreteVariables
        )

    def PrintIterPointInfo(self, savedNewPoints: SearchDataItem):
        point = savedNewPoints[0].GetY().floatVariables
        dpoint = savedNewPoints[0].GetY().discreteVariables
        value = savedNewPoints[0].GetZ()

        self.__functions.printIter(
            point,
            dpoint,
            value,
            self.iterNum, self.ndv
        )

        self.iterNum += 1

    def PrintBestPointInfo(self, solution, iters):
        if self.iterNum % iters != 0:
            pass
        else:
            bestTrialPoint = solution.bestTrials[0].point.floatVariables
            bestTrialDPoint = solution.bestTrials[0].point.discreteVariables
            bestTrialValue = solution.bestTrials[0].functionValues[0].value
            self.__functions.printBest(
                solution.numberOfGlobalTrials,
                solution.numberOfLocalTrials,
                solution.solutionAccuracy,
                bestTrialPoint,
                bestTrialDPoint,
                bestTrialValue,
                self.iterNum, self.ndv
            )
        self.iterNum += 1

    def PrintFinalResultInfo(self, solution: Solution, status: bool):
        bestTrialPoint = solution.bestTrials[0].point.floatVariables
        bestTrialDPoint = solution.bestTrials[0].point.discreteVariables
        bestTrialValue = solution.bestTrials[0].functionValues[0].value
        self.__functions.printResult(
            status,
            solution.numberOfGlobalTrials,
            solution.numberOfLocalTrials,
            solution.solvingTime,
            solution.solutionAccuracy,
            bestTrialPoint,
            bestTrialDPoint,
            bestTrialValue, self.ndv
        )


class OutputFunctions:

    def printInit(self, eps, r, epsR, itersLimit, floatdim, numberOfObjectives, numberOfConstraints,
                  lowerBoundOfFloatVariables, upperBoundOfFloatVariables, numberOfDisreteVariables):
        dim = floatdim + numberOfDisreteVariables
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

    def printIter(self, point, dpoint, value, iter, ndv):
        dim1 = len(point)
        dim2 = len(dpoint)
        print("|", end=' ')
        # print("\033[A|", end=' ')
        print("{:>5}:".format(iter), end=' ')
        print("{:>19.8f}".format(value), end='   ')
        if ndv > 0:
            print("{:<{width}}|".format(str(point) + " with " + str(dpoint), width = 20 * (dim1 + dim2)))
        else:
            print("{:<{width}}|".format(str(point), width=20 * dim1))

    def printResult(self, solved, numberOfGlobalTrials, numberOfLocalTrials, solvingTime, solutionAccuracy,
                    bestTrialPoint, bestTrialDPoint, bestTrialValue, ndv):
        dim = len(bestTrialPoint) + len(bestTrialDPoint)
        print("-" * (30 + 20 * dim + 2))
        print("|{:^{width}}|".format("Result", width=30 + 20 * dim))
        print("-" * (30 + 20 * dim + 2))
        # print("|{:>29} {:<{width}}|".format("is solved: ", str(solved), width=20*dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", numberOfGlobalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", numberOfLocalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("solving time: ", solvingTime, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("solution point: ", str(bestTrialPoint), width=20 * dim))
        if ndv > 0:
            print("|{:>29} {:<{width}}|".format("best disrete combination: ", str(bestTrialDPoint), width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("solution value: ", bestTrialValue, width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("accuracy: ", solutionAccuracy, width=20 * dim))
        print("-" * (30 + 20 * dim + 2))

    def printBest(self, numberOfGlobalTrials, numberOfLocalTrials, solutionAccuracy,
                  bestTrialPoint, bestTrialDPoint, bestTrialValue, iter, ndv):
        dim = len(bestTrialPoint) + len(bestTrialDPoint)
        print("|{:>29} {:<{width}}|".format("current iteration # ", iter, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", numberOfGlobalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", numberOfLocalTrials, width=20 * dim))
        print("|{:>29} {:<{width}}|".format("current best point: ", str(bestTrialPoint), width=20 * dim))
        if ndv > 0:
            print("|{:>29} {:<{width}}|".format("with disrete combination: ", str(bestTrialDPoint), width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("current best value: ", bestTrialValue, width=20 * dim))
        print("|{:>29} {:<{width}.8f}|".format("currant accuracy: ", solutionAccuracy, width=20 * dim))
        print("." * (30 + 20 * dim + 2))
