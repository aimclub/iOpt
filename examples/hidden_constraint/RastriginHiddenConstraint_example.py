from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.rastrigin_hidden_constraint import RastriginHiddenConstraint

if __name__ == "__main__":
    """
    Минимизация тестовой функции Растригина
    """

    # Создание тестовой задачи
    problem = RastriginHiddenConstraint(2)
    # Параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, itersLimit=3000, refineSolution=True)
    # Создание решателя
    solver = Solver(problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleOutputListener(mode='result')
    solver.AddListener(cfol)
    # Запуск решения задачи
    sol = solver.Solve()
