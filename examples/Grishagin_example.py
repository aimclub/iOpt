from problems.grishagin import Grishagin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Гришагина с визуализацией
    """

    # создание объекта задачи
    problem = Grishagin(function_number=1)

    # Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.AddListener(cfol)

    # Добавляем построение 3D визуализации после решения задачи
    spl = StaticPainterNDListener("grishagin.png", "output", varsIndxs=[0, 1], mode="lines layers",
                                calc="objective function")
    solver.AddListener(spl)

    # Решение задачи
    sol = solver.Solve()
