from problems.shekel4 import Shekel4
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Шекеля (размерность = 4)
    """

    # создание объекта задачи
    problem = Shekel4(function_number=1)

    # Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, iters_limit=10000, refine_solution=True)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Решение задачи
    sol = solver.solve()
