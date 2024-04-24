import traceback
from datetime import datetime

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.async_calculator import AsyncCalculator
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.calculator import Calculator


class AsyncParallelProcess(Process):
    def __init__(
        self,
        parameters: SolverParameters,
        task: OptimizationTask,
        evolvent: Evolvent,
        search_data: SearchData,
        method: Method,
        listeners: list[Listener],
        calculator: Calculator = None
    ):
        super(AsyncParallelProcess, self).__init__(
            parameters, task, evolvent, search_data, method, listeners, calculator
        )
        from iOpt.method.solverFactory import SolverFactory
        self.calculator = AsyncCalculator(SolverFactory.create_evaluate_method(task), parameters)

    def do_global_iteration(self, number: int = 1) -> None:
        done_trials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.before_method_start(self.method)
            done_trials = self.method.first_iteration()
            self._first_iteration = False
            number -= 1

        for _ in range(number):
            for _ in range(self.calculator.waiting_workers):
                newpoint, oldpoint = self.method.calculate_iteration_point()
                self.calculator.give_point(newpoint, oldpoint)
                self.method.finalize_iteration()

            for newpoint, oldpoint in self.calculator.take_list_of_calculated_points():
                self.method.update_optimum(newpoint)
                self.method.renew_search_data(newpoint, oldpoint)

            done_trials.extend(
                self.search_data.get_last_items(self.calculator.waiting_workers)
            )

        for listener in self._listeners:
            listener.on_end_iteration(done_trials, self.get_results())

    def solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: Текущая оценка решения задачи оптимизации
        """
        self.calculator.start()

        start_time = datetime.now()
        try:
            while not self.method.check_stop_condition():
                self.do_global_iteration()
        except Exception:
            print("Exception was thrown")
            print(traceback.format_exc())

        for newpoint, oldpoint in self.calculator.stop():
            self.method.update_optimum(newpoint)
            self.method.renew_search_data(newpoint, oldpoint)

        if self.parameters.refine_solution:
            self.do_local_refinement(self.parameters.local_method_iteration_count)

        result = self.get_results()
        result.solving_time += (datetime.now() - start_time).total_seconds()

        for listener in self._listeners:
            status = self.method.check_stop_condition()
            listener.on_method_stop(self.search_data, self.get_results(), status)

        return result
