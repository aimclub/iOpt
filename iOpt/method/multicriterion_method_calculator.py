from iOpt.method.index_method_calculator import IndexMethodCalculator
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem


class MulticriterionMethodCalculator(IndexMethodCalculator):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self, task: OptimizationTask):
        super().__init__(task)

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        # из IndexMethod
        # Вычисляются ВСЕ критерии
        # Добавить вычисление свертки

        pass

#сделать фабрику для MethodCalculator и вызвать ее в ParallelProcess