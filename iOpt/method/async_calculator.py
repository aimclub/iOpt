import multiprocess as mp

from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.search_data import SearchDataItem
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


class AsyncCalculator:
    def __init__(
        self, evaluate_method: ICriterionEvaluateMethod, parameters: SolverParameters
    ):
        self.evaluate_method = evaluate_method
        self.task_queue = mp.Queue()
        self.done_queue = mp.Queue()
        self.workers = [
            Worker(evaluate_method, self.task_queue, self.done_queue)
            for _ in range(parameters.number_of_parallel_points)
        ]
        self.waiting_workers = parameters.number_of_parallel_points
        self.waiting_oldpoints: dict[float, SearchDataItem] = dict()

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def give_point(self, newpoint: SearchDataItem, oldpoint: SearchDataItem) -> None:
        self.task_queue.put_nowait(newpoint)
        self.waiting_oldpoints[newpoint.get_x()] = oldpoint
        oldpoint.blocked = True

    def _take_calculated_point(
        self, block: bool
    ) -> tuple[SearchDataItem, SearchDataItem]:
        newpoint = self.done_queue.get(block=block)
        self.evaluate_method.copy_functionals(newpoint, newpoint)
        oldpoint = self.waiting_oldpoints.pop(newpoint.get_x())
        oldpoint.blocked = False
        return newpoint, oldpoint

    def take_list_of_calculated_points(
        self
    ) -> list[tuple[SearchDataItem, SearchDataItem]]:
        list_points = []
        points = self._take_calculated_point(block=True)
        list_points.append(points)
        self.waiting_workers = 1
        while not self.done_queue.empty():
            points = self._take_calculated_point(block=False)
            list_points.append(points)
            self.waiting_workers += 1
        return list_points

    def stop(self) -> list[tuple[SearchDataItem, SearchDataItem]]:
        for _ in range(len(self.workers)):
            self.task_queue.put_nowait("STOP")
        for w in self.workers:
            w.join()
        list_points = []
        while not self.done_queue.empty():
            points = self._take_calculated_point(block=False)
            list_points.append(points)
        return list_points

    def calculate_functionals_for_items(
        self, points: list[SearchDataItem]
    ) -> list[SearchDataItem]:
        for point in points:
            self.task_queue.put_nowait(point)
        points_res = []
        for _ in range(len(points)):
            points_res.append(self.done_queue.get())
        points_res.sort(key=lambda p: p.get_x())
        for point, point_r in zip(points, points_res):
            self.evaluate_method.copy_functionals(point, point_r)
        return points
