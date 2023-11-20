from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.GKLS import GKLS

if __name__ == "__main__":
    """
    Минимизация тестовой функции из GKLS генератора с номером 39
    """

    # создание объекта задачи
    problem = GKLS(dimension=6, functionNumber=39)

    # Формируем параметры решателя
    params = SolverParameters(r=7, eps=0.01, iters_limit=3000000,
                              number_of_parallel_points=4, timeout=1)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='result')
    solver.add_listener(cfol)

    # Решение задачи
    solver.solve()
