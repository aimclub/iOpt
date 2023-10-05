import multiprocessing as mp
import traceback
from datetime import datetime

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
        self.task_queue = mp.Queue()
        self.done_queue = mp.Queue()
        self.workers = [
            Worker(self.index_method_calculator, self.task_queue, self.done_queue)
            for _ in range(self.parameters.number_of_parallel_points)
        ]
        self.waiting_workers = self.parameters.number_of_parallel_points
        self.waiting_oldpoints: dict[float, SearchDataItem] = dict()

    def do_global_iteration(self, number: int = 1):
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
                self.task_queue.put_nowait(newpoint)
                self.waiting_oldpoints[newpoint.get_x()] = oldpoint
                # print(f"\tNew {newpoint.get_x()} at ({oldpoint.get_left().get_x()}, {oldpoint.get_x()})")
                # print(f'\t{newpoint.get_x()} is in {newpoint.get_x() in self.waiting_oldpoints}')
            # print(f"\tWait {self.waiting_oldpoints.keys()}")
            newpoint = self.done_queue.get()
            oldpoint = self.waiting_oldpoints.pop(newpoint.get_x())
            # print(f"\n\tNew with func value {newpoint.get_x()}")
            # oldpoint = self.search_data.find_data_item_by_one_dimensional_point(newpoint.get_x())
            self.method.update_optimum(newpoint)
            self.method.renew_search_data(newpoint, oldpoint)
            self.waiting_workers = 1
            while not self.done_queue.empty():
                newpoint = self.done_queue.get()
                oldpoint = self.waiting_oldpoints.pop(newpoint.get_x())
                # print(f"\tNew with func value {newpoint.get_x()}")
                # oldpoint = self.search_data.find_data_item_by_one_dimensional_point(newpoint.get_x())
                self.method.update_optimum(newpoint)
                self.method.renew_search_data(newpoint, oldpoint)
                self.waiting_workers += 1
            # print(f'\tStill wait {self.waiting_oldpoints.keys()}')
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

    def start_workers(self):
        for w in self.workers:
            w.start()

    def stop_workers(self):
        for _ in range(len(self.workers)):
            self.task_queue.put_nowait("STOP")
        for w in self.workers:
            w.join()
        while not self.done_queue.empty():
            newpoint = self.done_queue.get()
            oldpoint = self.waiting_oldpoints.pop(newpoint.get_x())
            self.method.update_optimum(newpoint)
            self.method.renew_search_data(newpoint, oldpoint)
            self.method.finalize_iteration()

    # def find_oldpoint(self, point: SearchDataItem) -> SearchDataItem:
    #     for i, oldpoint in enumerate(self.waiting_oldpoints):
    #         if oldpoint.get_left().get_x() < point.get_x() < oldpoint.get_x():
    #             return self.waiting_oldpoints.pop(i)
