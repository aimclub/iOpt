from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point
from problems.rastriginInt import RastriginInt

if __name__ == "__main__":
    """
    Минимизация тестовой функции Растригина с визуализацией
    """

    # Создание тестовой задачи
    problem = RastriginInt(dimension=5, number_of_discrete_variables=3)
    # Начальная точка
    start_point: Point = Point(float_variables=[0.5, 0.5], discrete_variables=['A', 'B', 'A'])
    # Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, iters_limit=10000, start_point=start_point, number_of_parallel_points=16)
    # Создание решателя
    solver = Solver(problem=problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    # 3D визуализация по окончании решения задачи

    # Запуск решения задачи
    sol = solver.solve()
