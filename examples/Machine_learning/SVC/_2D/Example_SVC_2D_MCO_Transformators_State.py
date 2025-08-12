from iOpt.output_system.listeners.static_painters import StaticPainterParetoListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import MCO_SVC_2D_Transformators_State
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import csv

# Для построения HV индекса
# from pymoo.util.misc import stack
# from pymoo.indicators.hv import HV
# import numpy as np

def factory_dataset():
    x = []
    y = []
    with open(r"../Datasets/transformator_state.csv") as rrrr_file:
        file_reader = csv.reader(rrrr_file, delimiter=",")
        for row in file_reader:
            x_row = []
            for i in range(len(row)-1):
                x_row.append(row[i])
            x.append(x_row)
            y.append(row[len(row)-1])
    return shuffle(np.array(x), np.array(y), random_state=42)


if __name__ == "__main__":
    X, Y = factory_dataset()
    regularization_value_bound = {'low': 5, 'up': 9}
    kernel_coefficient_bound = {'low': -3, 'up': 1}
    problem = MCO_SVC_2D_Transformators_State.MCO_SVC_2D_Transformators_State(X, Y, regularization_value_bound,
                                                                      kernel_coefficient_bound)
    method_params = SolverParameters(r=np.double(2.0), iters_limit=500, number_of_parallel_points=10,
                                     evolvent_density=12, number_of_lambdas=5)
    solver = Solver(problem=problem, parameters=method_params)
    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    sppl = StaticPainterParetoListener("MCO_SVC_2D_Transformators_pareto.png")
    solver.add_listener(sppl)

    # Решаем задачу
    sol = solver.solve()

    # Выводим множество Парето (координаты - значения функций)
    var = [trial.point.float_variables for trial in sol.best_trials]
    val = [[trial.function_values[i].value for i in range(2)] for trial in sol.best_trials]

    print("size pareto set: ", len(var))
    for fvar, fval in zip(var, val):
        print(fvar, fval)

    # Строим график множества Парето z[0]-z[1]
    fv1 = [trial.function_values[0].value for trial in sol.best_trials]
    fv2 = [-trial.function_values[1].value for trial in sol.best_trials]
    plt.plot(fv1, fv2, 'ro')
    plt.show()

    ########################################################################
    ## Вычисляем HV индекс
    ########################################################################

    ## Вычисляем отраженное Парето множество
    # data = 1-np.array(val)
    # ref_point = np.array([1.0, 1.0])
    # print("reference point:")
    # print(ref_point)
    # print("Pareto set:")
    # print(data)

    ## Создаем объект для подсчета HV индекса
    # ind = HV(ref_point=ref_point)

    ## Считаем и выводим HV индекс
    # print("HV", ind(data))