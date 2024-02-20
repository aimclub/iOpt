from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.XGBoostRegression._2D.Problems.XGBR_2D_Gasturbine import XGBR_2d_Gasturbine
from sklearn.utils import shuffle
import numpy as np
import csv

def gasturbine_Dataset():
    x = []
    y = []
    with open(r"../Datasets/no_predict.csv") as rrrr_file:
        file_reader = csv.reader(rrrr_file, delimiter=";")
        for row in file_reader:
            x_row = []
            for i in range(len(row)-1):
                x_row.append(row[i])
            x.append(x_row)
            y.append(row[len(row)-1])
    return shuffle(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), random_state=42)


if __name__ == "__main__":
    X, Y = gasturbine_Dataset()
    learning_rate_bound = {'low': 0.2, 'up': 0.4}
    gamma_bound = {'low': 0.2, 'up': 0.3}
    problem = XGBR_2d_Gasturbine(X, Y, learning_rate_bound, gamma_bound)
    method_params = SolverParameters(r=np.double(2.0), iters_limit=1000, number_of_parallel_points=12,
                                     evolvent_density=12)
    solver = Solver(problem=problem, parameters=method_params)
    spl1 = StaticPainterNDListener("gas_regr.png", "output", vars_indxs=[0, 1], mode="surface",
                                  calc="by points")
    solver.add_listener(spl1)
    spl2 = StaticPainterNDListener("gas_regr2.png", "output", vars_indxs=[0, 1], mode="lines layers",
                                  calc="by points")
    solver.add_listener(spl2)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()