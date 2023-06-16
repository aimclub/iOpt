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
    problem = RastriginInt(dimension=5, numberOfDiscreteVariables=3)
    # Начальная точка
    startPoint: Point = Point(floatVariables=[0.5, 0.5], discreteVariables=['A', 'B', 'A'])
    # Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=10000, startPoint=startPoint, numberOfParallelPoints=16)
    # Создание решателя
    solver = Solver(problem=problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleOutputListener(mode='full')
    solver.AddListener(cfol)
    # 3D визуализация по окончании решения задачи

    # Запуск решения задачи
    sol = solver.Solve()

