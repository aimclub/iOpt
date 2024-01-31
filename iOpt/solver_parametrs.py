import numpy as np
from iOpt.trial import Point


class SolverParameters:
    """
    Класс SolverParameters позволяет определить параметры поиска оптимального решения
    """

    def __init__(self,
                 eps: np.double = 0.01,
                 r: np.double = 2.0,
                 iters_limit: int = 20000,
                 evolvent_density: int = 10,
                 eps_r: np.double = 0.01,
                 refine_solution: bool = False,
                 start_point: Point = [],
                 number_of_parallel_points: int = 1,
                 timeout: int = -1,
                 proportion_of_global_iterations: float = 0.95,
                 start_lambdas: list = [],
                 number_of_lambdas: int = 10,
                 is_scaling: bool = False
                 ):
        r"""
        Конструктор класса SolverParameters

        :param eps: Точность решения поставленной задачи. Меньше значения -- выше точность поиска,
             меньше вероятность преждевременной остановки.
        :param r: Параметр надежности. Более высокое значение r -- более медленная сходимость,
             более высокая вероятность нахождения глобального минимума.
        :param iters_limit: максимальное число поисковых испытаний.
        :param evolvent_density: плотность построения развертки.
             По умолчанию плотность :math:`2^{-10}` на гиперкубе :math:`[0,1]^N`,
             что означает, что максимальная точность поиска составляет :math:`2^{-10}`.
        :param eps_r: параметр, влияющий на скорость решения задачи с ограничениями. eps_r = 0 - медленная сходимоть
             к точному решению, eps_r>0 - быстрая сходимть в окрестность решения.
        :param refine_solution: если true, то решение будет уточнено с помощью локального метода.
        :param start_point: точка начального приближения к решению.
        :param number_of_parallel_points: число параллельно вычисляемых испытаний.
        :param timeout: ограничение на время вычислений в минутах.
        :param proportion_of_global_iterations: доля глобальных итераций в поиске при использовании локальном метода
        """
        self.eps = eps
        self.r = r
        self.iters_limit = iters_limit
        self.proportion_of_global_iterations = proportion_of_global_iterations
        if refine_solution:
            self.global_method_iteration_count = int(self.iters_limit * self.proportion_of_global_iterations)
            self.local_method_iteration_count = self.iters_limit - self.global_method_iteration_count
        else:
            self.global_method_iteration_count = self.iters_limit
            self.local_method_iteration_count = 0

        self.evolvent_density = evolvent_density
        self.eps_r = eps_r
        self.refine_solution = refine_solution
        self.start_point = start_point
        self.number_of_parallel_points = number_of_parallel_points
        self.timeout = timeout

        self.start_lambdas = start_lambdas # тут бы проверку, что они в сумме дают 1 и что их нужное количество
        self.number_of_lambdas = number_of_lambdas
        self.is_scaling = is_scaling
