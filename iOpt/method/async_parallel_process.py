import traceback
from datetime import datetime

import multiprocess as mp

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.index_method_calculator import IndexMethodCalculator
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


class Worker(mp.Process):
    def __init__(
        self,
        evaluate_method: ICriterionEvaluateMethod,
        task_queue: mp.Queue,
        done_queue: mp.Queue,
    ):
        super(Worker, self).__init__()
        self.evaluate_method = evaluate_method
        self.task_queue = task_queue
        self.done_queue = done_queue

    def run(self):
        for point in iter(self.task_queue.get, "STOP"):
            point = self.evaluate_method.calculate_functionals(point)
            self.done_queue.put_nowait(point)


class AsyncParallelProcess(Process):
    def __init__(
        self,
        parameters: SolverParameters,
        task: OptimizationTask,
        evolvent: Evolvent,
        search_data: SearchData,
        method: Method,
        listeners: list[Listener],
    ):
        super(AsyncParallelProcess, self).__init__(
            parameters, task, evolvent, search_data, method, listeners
        )
        self.index_method_calculator = IndexMethodCalculator(task)
        # self.calculator = Calculator(self.index_method_calculator, parameters)
        self.task_queue = mp.Queue()
        self.done_queue = mp.Queue()
        self.workers = [
            Worker(self.index_method_calculator, self.task_queue, self.done_queue)
            for _ in range(self.parameters.number_of_parallel_points)
        ]
        self.waiting_workers = self.parameters.number_of_parallel_points
        self.waiting_oldpoints: dict[float, SearchDataItem] = dict()

    def start_workers(self) -> None:
        for w in self.workers:
            w.start()

    def give_point_to_workers(
        self, newpoint: SearchDataItem, oldpoint: SearchDataItem
    ) -> None:
        self.task_queue.put_nowait(newpoint)
        self.waiting_oldpoints[newpoint.get_x()] = oldpoint
        oldpoint.blocked = True

    def take_point_from_workers(
        self, block: bool
    ) -> tuple[SearchDataItem, SearchDataItem]:
        newpoint = self.done_queue.get(block=block)
        oldpoint = self.waiting_oldpoints.pop(newpoint.get_x())
        oldpoint.blocked = False
        return newpoint, oldpoint

    def take_list_points_from_workers(
        self
    ) -> list[tuple[SearchDataItem, SearchDataItem]]:
        list_points = []
        points = self.take_point_from_workers(block=True)
        list_points.append(points)
        self.waiting_workers = 1
        while not self.done_queue.empty():
            points = self.take_point_from_workers(block=False)
            list_points.append(points)
            self.waiting_workers += 1
        return list_points

    def stop_workers(self) -> None:
        for _ in range(len(self.workers)):
            self.task_queue.put_nowait("STOP")
        for w in self.workers:
            w.join()
        while not self.done_queue.empty():
            newpoint, oldpoint = self.take_point_from_workers(block=False)
            self.method.update_optimum(newpoint)
            self.method.renew_search_data(newpoint, oldpoint)

    def do_global_iteration(self, number: int = 1) -> None:
        done_trials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.before_method_start(self.method)
            done_trials = self.method.first_iteration()
            self._first_iteration = False
            number -= 1

        for _ in range(number):
            for _ in range(self.waiting_workers):
                newpoint, oldpoint = self.method.calculate_iteration_point()
                self.give_point_to_workers(newpoint, oldpoint)

            for newpoint, oldpoint in self.take_list_points_from_workers():
                self.method.update_optimum(newpoint)
                self.method.renew_search_data(newpoint, oldpoint)
            self.method.finalize_iteration()
            done_trials.extend(self.search_data.get_last_items(self.waiting_workers))

        for listener in self._listeners:
            listener.on_end_iteration(done_trials, self.get_results())

    def solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: Текущая оценка решения задачи оптимизации
        """
        self.start_workers()

        start_time = datetime.now()
        try:
            while not self.method.check_stop_condition():
                self.do_global_iteration()
        except Exception:
            print("Exception was thrown")
            print(traceback.format_exc())

        self.stop_workers()

        if self.parameters.refine_solution:
            self.do_local_refinement(self.parameters.local_method_iteration_count)

        result = self.get_results()
        result.solving_time = (datetime.now() - start_time).total_seconds()

        for listener in self._listeners:
            status = self.method.check_stop_condition()
            listener.on_method_stop(self.search_data, self.get_results(), status)

        return result
