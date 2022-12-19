from iOpt.problems.shekel import Shekel
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener, ConsoleFullOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Шекеля
    """

    # создание объекта задачи
    problem = Shekel(0)

    # Формируем параметры решателя
    params = SolverParameters(r=3, eps=0.01, itersLimit=300, refineSolution=True)

    # Создаем решатель
    solver = Solver(problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)

    # Добавляем построение визуализации после решения задачи
    spl = StaticPaintListener("shekel.png", "output", indx=0, isPointsAtBottom=False, mode="objective function")
    solver.AddListener(spl)

    # Решение задачи
    sol = solver.Solve()
