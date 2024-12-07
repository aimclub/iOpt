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
        self.number_of_constraints = self.problem.number_of_constraints

    def print_init_info(self):
        self.__functions.print_init(
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

    def print_iter_point_info(self, saved_new_points: List[SearchDataItem]):
        if self.parameters.number_of_parallel_points > 1:
            isFirst = True
        else:
            isFirst = False

        for i in range(len(saved_new_points)):
            point = saved_new_points[i].get_y().float_variables
            dpoint = saved_new_points[i].get_y().discrete_variables
            value = saved_new_points[i].get_z()

            self.__functions.print_iter(
                point,
                dpoint,
                value,
                self.iterNum, self.ndv,
                isFirst
            )
            isFirst = False

            self.iterNum += 1

    def print_best_point_info(self, solution, iters):
        if self.iterNum % iters != 0:
            pass
        else:
            best_trial_point = solution.best_trials[0].point.float_variables
            best_trial_d_point = solution.best_trials[0].point.discrete_variables
            best_trial_value = solution.best_trials[0].function_values[self.number_of_constraints].value
            self.__functions.print_best(
                solution.number_of_global_trials,
                solution.number_of_local_trials,
                solution.solution_accuracy,
                best_trial_point,
                best_trial_d_point,
                best_trial_value,
                self.iterNum, self.ndv
            )
        self.iterNum += 1

    def print_final_result_info(self, solution: Solution, status: bool):
        best_trial_point = solution.best_trials[0].point.float_variables
        best_trial_d_point = solution.best_trials[0].point.discrete_variables
        best_trial_value = solution.best_trials[0].function_values[self.number_of_constraints].value
        self.__functions.print_result(
            status,
            solution.number_of_global_trials,
            solution.number_of_local_trials,
            solution.solving_time,
            solution.solution_accuracy,
            best_trial_point,
            best_trial_d_point,
            best_trial_value, self.ndv
        )

    def print_pareto_set_info(self, solution: Solution):
        self.__functions.print_pareto_set(solution.best_trials)

class OutputFunctions:

    def print_init(self, eps, r, eps_r, iters_limit, floatdim, number_of_objectives, number_of_constraints,
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
        print("|{:>29} {:<{width}}|".format("objective-function count: ", number_of_objectives,
                                            width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("constraint-function count: ", number_of_constraints,
                                            width=size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Method Parameters", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:>29} {:<{width}}|".format("eps: ", eps, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("r: ", r, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("eps_r: ", eps_r, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("iters_limit: ", iters_limit, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("number_of_parallel_points: ", number_of_parallel_points,
                                            width=size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Iterations", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("", width=30 + size_max_one_output * dim))

    def print_iter(self, point, dpoint, value, iter, ndv, flag):
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
            print("{:<{width}}|".format(str(point) + " with " + str(dpoint), width=size_max_one_output * (dim1 + dim2)))
        else:
            print("{:<{width}}|".format(str(point), width=size_max_one_output * dim1))

    def print_result(self, solved, number_of_global_trials, number_of_local_trials, solving_time,
                     solution_accuracy, best_trial_point, best_trial_d_point, best_trial_value, ndv):
        size_max_one_output = 15
        dim = len(best_trial_point) + len(best_trial_d_point)
        print("-" * (30 + size_max_one_output * dim + 2))
        print("|{:^{width}}|".format("Result", width=30 + size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))
        # print("|{:>29} {:<{width}}|".format("is solved: ", str(solved), width=20*dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", number_of_global_trials,
                                            width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", number_of_local_trials,
                                            width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("solving time: ", solving_time, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("solution point: ", str(best_trial_point), width=size_max_one_output * dim))
        if ndv > 0:
            print("|{:>29} {:<{width}}|".format("best disrete combination: ", str(best_trial_d_point),
                                                width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("solution value: ", best_trial_value, width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("accuracy: ", solution_accuracy, width=size_max_one_output * dim))
        print("-" * (30 + size_max_one_output * dim + 2))

    def print_best(self, number_of_global_trials, number_of_local_trials, solution_accuracy,
                   best_trial_point, best_trial_d_point, best_trial_value, curr_iter, ndv):
        size_max_one_output = 15
        dim = len(best_trial_point) + len(best_trial_d_point)
        print("|{:>29} {:<{width}}|".format("current iteration # ", curr_iter,
                                            width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("global iteration count: ", number_of_global_trials,
                                            width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("local iteration count: ", number_of_local_trials,
                                            width=size_max_one_output * dim))
        print("|{:>29} {:<{width}}|".format("current best point: ", str(best_trial_point),
                                            width=size_max_one_output * dim))
        if ndv > 0:
            print("|{:>29} {:<{width}}|".format("with discrete combination: ", str(best_trial_d_point),
                                                width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("current best value: ", best_trial_value,
                                               width=size_max_one_output * dim))
        print("|{:>29} {:<{width}.8f}|".format("currant accuracy: ", solution_accuracy,
                                               width=size_max_one_output * dim))
        print("." * (30 + size_max_one_output * dim + 2))

    def print_pareto_set(self, best_trials):
        size_max_one_output = 15
        dim = len(best_trials[0].point.float_variables)
        criteria_count = len(best_trials[0].function_values)

        var = [trial.point.float_variables for trial in best_trials]
        val = [[trial.function_values[i].value for i in range(len(trial.function_values))] for trial in best_trials]

        string_len = size_max_one_output * (dim + criteria_count) + 6
        print("| {:^{width}} |".format(f"Size pareto set: {len(var)}", width=string_len))
        print("-" * (string_len + 4))

        for fvar, fval in zip(var, val):
            print("| {:>{width_point}}".format(str(fvar), width_point=dim * size_max_one_output), end='  ')
            print("{:<{width_criteria}} |".format(str(fval), width_criteria=criteria_count * size_max_one_output))

        print("-" * (string_len + 4))
