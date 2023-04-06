from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method
import numpy as np

class Listener:
    """
    Базовый класс слушателя событий.
    """

    def BeforeMethodStart(self, method: Method):
        pass

    def OnEndIteration(self, currPoints : np.ndarray(shape=(1), dtype=SearchDataItem), solution: Solution):
        pass

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool):
        pass

    def OnRefrash(self):
        pass
