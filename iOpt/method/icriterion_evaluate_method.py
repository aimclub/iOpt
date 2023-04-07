from abc import ABC, abstractmethod

from iOpt.method.search_data import SearchDataItem


class ICriterionEvaluateMethod (ABC):

    @abstractmethod
    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        pass

    @abstractmethod
    def CopyFunctionals(self, dist_point: SearchDataItem, src_point: SearchDataItem):
        pass
