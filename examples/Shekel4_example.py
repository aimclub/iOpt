from iOpt.problems.shekel4 import Shekel4
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import ConsoleFullOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Шекеля (размерность = 4)
    """

    # создание объекта задачи
    problem = Shekel4(1)

    # Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=10000, refineSolution=True)

    # Создаем решатель
    solver = Solver(problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)

    # Решение задачи
    sol = solver.Solve()
