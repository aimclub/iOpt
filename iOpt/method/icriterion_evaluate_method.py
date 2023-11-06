from abc import ABC, abstractmethod

from iOpt.method.search_data import SearchDataItem


class ICriterionEvaluateMethod(ABC):

    @abstractmethod
    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        pass

    @abstractmethod
    def copy_functionals(self, dist_point: SearchDataItem, src_point: SearchDataItem):
        pass
