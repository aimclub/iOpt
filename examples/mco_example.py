from problems.mco_test1 import mco_test1
from problems.mco_test3 import mco_test3
from problems.mco_test5 import mco_test5
from problems.mco_test6 import mco_test6
from iOpt.trial import Point
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.output_system.listeners.static_painters import StaticPainterNDListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Хилла c визуализацией
    """

    # создание объекта задачи
    problem = mco_test6()


    # Формируем параметры решателя
    params = SolverParameters(r=3, eps=0.01, iters_limit=3000)#, number_of_parallel_points=8)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Добавляем построение визуализации после решения задачи
    spl = StaticPainterNDListener(file_name="GKLS.png", path_for_saves="output", vars_indxs=[0, 1], mode="lines layers",
                                  calc="objective function")
    solver.add_listener(spl)

    # Решение задачи
    sol = solver.solve()
