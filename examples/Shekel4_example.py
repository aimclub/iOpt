import math
import unittest
import sys
import numpy as np

from iOpt.problems.shekel4 import Shekel4
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Минимизация тестовой функции Шекеля с визуализацией
    """

    #создание объекта задачи
    problem = Shekel4(1)

    #Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)

    #Создаем решатель
    solver = Solver(problem, parameters=params)

    #Добавляем вывод результатов в консоль
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)

    #Добавляем построение 3D визуализации после решения задачи
    spl = StaticNDPaintListener("shekel4.png", "output", varsIndxs=[0,1], mode="lines layers", calc="objective function")
    solver.AddListener(spl)

    #Решение задачи
    sol = solver.Solve()

