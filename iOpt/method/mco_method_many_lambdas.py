from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.mco_method import MCOMethod
from iOpt.method.mco_optim_task import MCOOptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters


class MCOMethodManyLambdas(MCOMethod):
    """
    The MCOMethodManyLambdas class contains an implementation of
    the Global Search Algorithm in the case of multiple convolutions
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: MCOOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator
                 ):
        super().__init__(parameters, task, evolvent, search_data, calculator)
        self.current_lambdas = None
        self.is_recalc_all_convolution = True
        self.max_iter_for_convolution = 0
        self.number_of_lambdas = parameters.number_of_lambdas

        if parameters.start_lambdas:
            self.start_lambdas = parameters.start_lambdas
        else:
            self.start_lambdas = []

        self.current_num_lambda = 0
        self.lambdas_list = []
        self.iterations_list = []

        self.convolution = task.convolution

        self.init_lambdas()

    def check_stop_condition(self) -> bool:
        if super().check_stop_condition():
            if self.current_num_lambda < self.number_of_lambdas:
                self.change_lambdas()
        return super().check_stop_condition()

    def change_lambdas(self) -> None:
        self.set_min_delta(1)
        self.current_num_lambda += 1
        if self.current_num_lambda < self.number_of_lambdas:
            self.current_lambdas = self.lambdas_list[self.current_num_lambda]
            self.task.convolution.lambda_param = self.current_lambdas

            self.iterations_list.append(self.iterations_count)
            max_iter_for_convolution = int((self.parameters.global_method_iteration_count /
                                            self.number_of_lambdas) * (self.current_num_lambda + 1))
            self.set_max_iter_for_convolution(max_iter_for_convolution)

    def init_lambdas(self) -> None:
        if self.task.problem.number_of_objectives == 2:
            if self.number_of_lambdas > 1:
                h = 1.0 / (self.number_of_lambdas - 1)
            else:
                h = 1
            if not self.start_lambdas:
                for i in range(self.number_of_lambdas):
                    lambda_0 = i * h
                    if lambda_0 > 1:
                        lambda_0 = lambda_0 - 1
                    lambda_1 = 1 - lambda_0
                    lambdas = [lambda_0, lambda_1]
                    self.lambdas_list.append(lambdas)
            elif len(self.start_lambdas) == self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            elif len(self.start_lambdas) == 1:
                self.lambdas_list.append(self.start_lambdas[0])
                for i in range(1, self.number_of_lambdas):
                    lambda_0 = self.start_lambdas[0][0] + i * h
                    if lambda_0 > 1:
                        lambda_0 = lambda_0 - 1
                    lambda_1 = 1 - lambda_0
                    lambdas = [lambda_0, lambda_1]
                    self.lambdas_list.append(lambdas)
        else:  # многомерный случай
            if len(self.start_lambdas) == self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            else:
                if self.number_of_lambdas > 1:
                    h = 1.0 / (self.number_of_lambdas - 1)
                else:
                    h = 1
                evolvent = Evolvent([0] * self.task.problem.number_of_objectives,
                                    [1] * self.task.problem.number_of_objectives,
                                    self.task.problem.number_of_objectives)

                for i in range(self.number_of_lambdas):
                    x = i * h
                    y = evolvent.get_image(x)
                    sum = 0
                    for i in range(self.task.problem.number_of_objectives):
                        sum += y[i]
                    for i in range(self.task.problem.number_of_objectives):
                        y[i] = y[i] / sum
                    lambdas = list(y)
                    self.lambdas_list.append(lambdas)

        self.current_lambdas = self.lambdas_list[0]
        self.max_iter_for_convolution = \
            int(self.parameters.global_method_iteration_count / self.number_of_lambdas)
