
from problems.stronginc3 import Stronginc3
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    """
    Минимизация тестовой функции Стронгина с тримя ограничениями
    """

    problem = Stronginc3()

    # Формируем параметры решателя
    params = SolverParameters(r=2.5, eps=0.01, iters_limit=500)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='result')
    solver.add_listener(cfol)

    # Решение задачи
    sol = solver.solve()

    val = problem.calculate(sol.best_trials[0].point, sol.best_trials[0].function_values[3])
