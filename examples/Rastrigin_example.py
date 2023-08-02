from iOpt.trial import Point
from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Растригина с визуализацией
    """

    # Создание тестовой задачи
    problem = Rastrigin(dimension=2)
    # Начальная точка
    start_point: Point = Point(float_variables=[0.5, 0.5], discrete_variables=None)
    # Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, iters_limit=300, refine_solution=True, start_point=start_point)
    # Создание решателя
    solver = Solver(problem=problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    # 3D визуализация по окончании решения задачи
    spl = StaticPainterNDListener(file_name="rastrigin.png", path_for_saves="output", vars_indxs=[0, 1], mode="surface",
                                  calc="interpolation")
    solver.add_listener(spl)
    # Запуск решения задачи
    sol = solver.solve()
