from problems.GKLS import GKLS
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции из GKLS генератора с номером 39
    """

    # создание объекта задачи
    problem = GKLS(dimension=2, functionNumber=39)

    # Формируем параметры решателя
    params = SolverParameters(r=3.5, eps=0.01, iters_limit=300, refine_solution=True, number_of_parallel_points=4)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Добавляем построение 3D визуализации после решения задачи
    spl = StaticPainterNDListener(file_name="GKLS.png", path_for_saves="output", vars_indxs=[0, 1], mode="lines layers",
                                  calc="objective function")
    solver.add_listener(spl)

    # Решение задачи
    solver.solve()
