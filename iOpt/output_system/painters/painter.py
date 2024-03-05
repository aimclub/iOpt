from iOpt.solution import Solution
from iOpt.method.search_data import SearchDataItem

class Painter:
    """
    Basic Drawer Class
    """
    def paint_objective_func(self):
        pass

    def paint_points(self, curr_point: SearchDataItem):
        pass

    def paint_optimum(self, solution: Solution):
        pass

    def save_image(self):
        pass