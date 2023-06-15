from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.GKLS import GKLS


def SolveSingleGKLS():
    """
    Минимизация тестовой функции из GKLS генератора с номером 39
    """

    # создание объекта задачи
    problem = GKLS(6, 39)

    # Формируем параметры решателя
    params = SolverParameters(r=7, eps=0.01, itersLimit=3000000,
                              numberOfParallelPoints=4, timeout=1)

    # Создаем решатель
    solver = Solver(problem, parameters=params)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='result')
    solver.AddListener(cfol)

    # Решение задачи
    sol = solver.Solve()


if __name__ == "__main__":
    SolveSingleGKLS()