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
    problem = GKLS(dimension=3, functionNumber=92)

    # Формируем параметры решателя
    params = SolverParameters(r=4, eps=0.01, refine_solution=False, number_of_parallel_points=8, async_scheme=True)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='result')
    solver.add_listener(cfol)

    # # Добавляем построение 3D визуализации после решения задачи
    # spl = StaticPainterNDListener(file_name="GKLS.png", path_for_saves="output", vars_indxs=[0, 1], mode="lines layers",
    #                               calc="objective function")
    # solver.add_listener(spl)

    # Решение задачи
    solver.solve()

