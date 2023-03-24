from abc import ABC, abstractmethod

from iOpt.method.search_data import SearchDataItem


class IMethod (ABC):

    @abstractmethod
    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        pass