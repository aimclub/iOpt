from iOpt.problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Минимизация тестовой функции Растригина с визуализацией
    """

    # Создание тестовой задачи
    problem = Rastrigin(2)
    # Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)
    # Создание решателя
    solver = Solver(problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)
    # 3D визуализация по окончании решения задачи
    spl = StaticNDPaintListener("rastrigin.png", "output", varsIndxs=[0, 1], mode="surface", calc="interpolation")
    solver.AddListener(spl)
    # Запуск решения задачи
    sol = solver.Solve()

