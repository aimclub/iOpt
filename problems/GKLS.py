from iOpt.problem import Problem
from problems.GKLS_function.gkls_function import GKLSClass, GKLSFuncionType, GKLSFunction
from iOpt.trial import Point, FunctionValue, Trial


class GKLS(Problem):
    """
    GKLS-генератор, позволяет порождать задачи многоэкстремальной оптимизации с заранее известными свойствами: 
    количеством локальных минимумов, размерами их областей притяжения, точкой глобального минимума, 
    значением функции в ней и т.п. 
    """

    def __init__(self, dimension: int,
                 functionNumber: int = 1) -> None:
        """
        Конструктор класса GKLS генератора. 

        :param dimension: Размерность задачи, :math:`2 <= dimension <= 5`
        :param functionNumber: номер задачи в наборе, :math:`1 <= functionNumber <= 100`
        """
        super(GKLS, self).__init__()
        self.dimension = dimension
        self.name = "GKLS"
        self.number_of_float_variables = dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = [str(x) for x in range(self.dimension)]

        self.lower_bound_of_float_variables = dimension * [-1]
        self.upper_bound_of_float_variables = dimension * [1]

        self.function: GKLSFunction = GKLSFunction()

        self.mMaxDimension: int = 5
        self.mMinDimension: int = 2

        self.function_number: int = functionNumber
        self.num_minima: int = 10

        self.problem_class: int = GKLSClass.Simple
        self.function_class: int = GKLSFuncionType.TD

        self.function.GKLS_global_value: float = -1.0
        self.function.NumberOfLocalMinima: int = self.num_minima
        self.function.SetDimension(self.dimension)
        self.function.mFunctionType: int = self.function_class

        self.function.SetFunctionClass(self.problem_class, self.dimension)

        self.global_dist: float = self.function.GKLS_global_dist
        self.global_radius: float = self.function.GKLS_global_radius

        self.function.GKLS_parameters_check()

        self.function.SetFunctionNumber(self.function_number)

        KOfunV = FunctionValue()
        KOfunV.value = self.function.GetOptimumValue()

        KOpoint = Point(self.function.GetOptimumPoint(), [])

        self.known_optimum = [Trial(KOpoint, [KOfunV])]

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения функции в заданной точке

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции

        :return: Вычисленное значение функции в точке point
        """
        function_value.value = self.function.Calculate(point.float_variables)
        return function_value
