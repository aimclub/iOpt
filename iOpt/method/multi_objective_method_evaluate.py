from iOpt.method.index_method_evaluate import IndexMethodEvaluate
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem
from iOpt.trial import FunctionValue, FunctionType
from iOpt.method.optim_task import TypeOfCalculation


class MultiObjectiveMethodEvaluate(IndexMethodEvaluate):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self, task: OptimizationTask):
        super().__init__(task)

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        # из IndexMethod
        # Вычисляются ВСЕ критерии
        # Добавить вычисление свертки

        number_of_constraints = self.task.problem.number_of_constraints
        for i in range(number_of_constraints):  # проходим по всем ограничениям
            point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)  # ???
            point = self.task.calculate(point,
                                        i)  # ??? типа считаем, что перестановок нет и индекс соответствует индексу ограничения?? И сначала ограничения, потом критерии
            point.set_z(point.function_values[i].value)
            point.set_index(i)
            if point.get_z() > 0:
                return point

        # Вычисляются ВСЕ критерии
        for i in range(self.task.problem.number_of_objectives):  # проходим по всем критериям
            point.function_values[number_of_constraints + i] = FunctionValue(FunctionType.OBJECTIV,
                                                                             number_of_constraints + i)
            point = self.task.calculate(point, number_of_constraints + i)

        # Добавить вычисление свертки
        point = self.task.calculate(point, -1, TypeOfCalculation.CONVOLUTION)
        point.set_index(number_of_constraints)



#сделать фабрику для MethodCalculator и вызвать ее в ParallelProcess