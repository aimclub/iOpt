import math
import unittest
import sys
import numpy as np

from iOpt.problems.grishagin import Grishagin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Минимизация тестовой функции Гришагина с визуализацией
    """

    #создание объекта задачи
    problem = Grishagin(1)

    #Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)

    #Создаем решатель
    solver = Solver(problem, parameters=params)

    #Добавляем вывод результатов в консоль
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)

    #Добавляем построение 3D визуализации после решения задачи
    spl = StaticNDPaintListener("grishagin.png", "output", varsIndxs=[0,1], mode="surface", calc="approximation")
    solver.AddListener(spl)

    #Решение задачи
    sol = solver.Solve()

