from iOpt.solution import Solution
from iOpt.method.search_data import SearchDataItem

class Painter:
    """
    Базовый класс рисовальщика.
    """
    def PaintObjectiveFunc(self):
        pass

    def PaintPoints(self, currPoint: SearchDataItem):
        pass

    def PaintOptimum(self, solution: Solution):
        pass

    def SaveImage(self):
        pass