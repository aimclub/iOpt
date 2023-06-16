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
    startPoint: Point = Point(floatVariables=[0.5, 0.5], discreteVariables=None)
    # Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True, startPoint=startPoint)
    # Создание решателя
    solver = Solver(problem=problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleOutputListener(mode='full')
    solver.AddListener(cfol)
    # 3D визуализация по окончании решения задачи
    spl = StaticPainterNDListener(fileName="rastrigin.png", pathForSaves="output", varsIndxs=[0, 1], mode="surface",
                                  calc="interpolation")
    solver.AddListener(spl)
    # Запуск решения задачи
    sol = solver.Solve()

