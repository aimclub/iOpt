from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.rastrigin_hidden_constraint import RastriginHiddenConstraint
from problems.rastrigin_int_hidden_constraint import RastriginIntHiddenConstraint

if __name__ == "__main__":
    """
    Минимизация тестовой функции Растригина
    """

    # Создание тестовой задачи
    problemInt = RastriginIntHiddenConstraint(2, 1)
    # Параметры решателя
    paramsInt = SolverParameters(r=2.5, eps=0.01, itersLimit=300, refineSolution=True)
    # Создание решателя
    solverInt = Solver(problemInt, parameters=paramsInt)
    # Вывод результатов в консоль в процессе решения задачи
    cfolInt = ConsoleOutputListener(mode='result')
    solverInt.AddListener(cfolInt)
    # Запуск решения задачи
    solInt = solverInt.Solve()
    print(solInt)
