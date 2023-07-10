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
    params = SolverParameters(r=2.5, eps=0.01, iters_limit=3000, refine_solution=True)
    # Создание решателя
    solver = Solver(problem, parameters=params)
    # Вывод результатов в консоль в процессе решения задачи
    cfol = ConsoleOutputListener(mode='result')
    solver.add_listener(cfol)
    # Запуск решения задачи
    sol = solver.solve()
