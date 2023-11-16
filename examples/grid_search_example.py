from problems.GKLS import GKLS
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Поиск на сетке тестовой функции из GKLS генератора с номером 39, сохранение и загрузка поисковой информации
    """

    # создание объекта задачи
    problem = GKLS(dimension=2, functionNumber=39)

    # Формируем параметры решателя
    params = SolverParameters(r=3.5, eps=0.0001, iters_limit=10000, number_of_parallel_points=4)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Решение задачи
    solver.grid_search()

    log = solver.save_progress()

    # Создаем решатель
    solver2 = Solver(problem=problem, parameters=params)

    # Добавляем построение 3D визуализации после решения задачи
    spl = StaticPainterNDListener(file_name="GKLS.png", path_for_saves="output", vars_indxs=[0, 1], mode="lines layers",
                                  calc="objective function")
    solver2.add_listener(spl)

    spl3d = StaticPainterNDListener(file_name="GKLS.png", path_for_saves="output", vars_indxs=[0, 1], mode="surface",
                                    calc="interpolation")
    solver2.add_listener(spl3d)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver2.add_listener(cfol)


    solver2.load_progress(log)

    solver2.release_all_listener()