from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.GKLS import GKLS


def solve_single_gkls():
    """
    Минимизация тестовой функции из GKLS генератора с номером 39
    """

    # создание объекта задачи
    problem = GKLS(dimension=3, functionNumber=39)

    # Формируем параметры решателя
    params = SolverParameters(r=4, eps=0.01, number_of_parallel_points=4, async_scheme=True)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Решение задачи
    solver.solve()


if __name__ == "__main__":
    solve_single_gkls()
