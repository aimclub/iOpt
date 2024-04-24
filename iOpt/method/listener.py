from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method
import numpy as np


class Listener:
    """
    Event listener base class
    """

    def before_method_start(self, method: Method):
        pass

    def on_end_iteration(self, curr_points: np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        pass

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        pass

    def on_refrash(self):
        pass
