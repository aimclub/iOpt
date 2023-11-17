from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import SVC_2D_Transformators_State
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import csv

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
    problem = SVC_2D_Transformators_State.SVC_2D_Transformators_State(X, Y, regularization_value_bound, kernel_coefficient_bound)
    method_params = SolverParameters(r=np.double(2.0), iters_limit=100)
    solver = Solver(problem, parameters=method_params)
    #apl = AnimatePainterNDListener("svc2d_transformator_state_anim.png", "output", vars_indxs=[0, 1])
    #solver.add_listener(apl)
    #spl = StaticPainterNDListener("svc2d_transformator_state_stat.png", "output", vars_indxs=[0, 1], mode="surface", calc="interpolation")
    #solver.add_listener(spl)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()
