import math
import unittest
import sys
import numpy as np

from iOpt.problems.stronginC3 import StronginC3
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Минимизация тестовой функции СтронгинС3 с визуализацией
    """

    #создание объекта задачи
    problem = StronginC3()

    #Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)

    #Создаем решатель
    solver = Solver(problem, parameters=params)

    #Добавляем вывод результатов в консоль
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)

    #Добавляем построение 3D визуализации после решения задачи
    spl = StaticNDPaintListener("shekel.png", "output", varsIndxs=[0,1], mode="surface", calc="interpolation")
    solver.AddListener(spl)

    #Решение задачи
    sol = solver.Solve()

