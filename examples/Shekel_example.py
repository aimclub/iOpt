from problems.shekel import Shekel
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Шекеля
    """

    # создание объекта задачи
    problem = Shekel(function_number=0)

    # Формируем параметры решателя
    params = SolverParameters(r=3, eps=0.01, itersLimit=300, refineSolution=True)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.AddListener(cfol)

    # Добавляем построение визуализации после решения задачи
    spl = StaticPainterListener(fileName="shekel.png", pathForSaves="output", indx=0, isPointsAtBottom=False,
                                mode="objective function")
    solver.AddListener(spl)

    # Решение задачи
    sol = solver.Solve()
