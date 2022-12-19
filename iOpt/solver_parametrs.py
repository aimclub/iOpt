import numpy as np
from iOpt.trial import Point


class SolverParameters:
    """
    Класс SolverParameters позволяет определить параметры поиска оптимального решения
    """
    def __init__(self,
                 eps: np.double = 0.01,
                 r: np.double = 2.0,
                 itersLimit: int = 20000,
                 evolventDensity: int = 10,
                 epsR: np.double = 0.001,
                 refineSolution: bool = False,
                 startPoint: Point = []
                 ):
        r"""
        Конструктор класса SolverParameters

        :param eps: Точность решения поставленной задачи. Меньше значения -- выше точность поиска,
             меньше вероятность преждевременной остановки.
        :param r: Параметр надежности. Более высокое значение r -- более медленная сходимость,
             более высокая вероятность нахождения глобального минимума.
        :param itersLimit: максимальное число поисковых испытаний.
        :param evolventDensity: плотность построения развертки.
             По умолчанию плотность :math:`2^{-10}` на гиперкубе :math:`[0,1]^N`,
             что означает, что максимальная точность поиска составляет :math:`2^{-10}`.
        :param epsR: параметр, влияющий на скорость решения задачи с ограничениями. epsR = 0 - медленная сходимоть
             к точному решению, epsR>0 - быстрая сходимть в окрестность решения.
        :param refineSolution: если true, то решение будет уточнено с помощью локального метода.
        :param startPoint: точка начального приближения к решению.
        """
        self.eps = eps
        self.r = r
        self.itersLimit = itersLimit
        self.evolventDensity = evolventDensity
        self.epsR = epsR
        self.refineSolution = refineSolution
        self.startPoint = startPoint
