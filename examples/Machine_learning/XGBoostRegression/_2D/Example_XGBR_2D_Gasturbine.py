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
    method_params = SolverParameters(r=np.double(2.0), iters_limit=200)
    solver = Solver(problem, parameters=method_params)
    apl = AnimatePainterNDListener("XGBR_2d_Gasturbine_anim.png", "output", vars_indxs=[0, 1])
    solver.add_listener(apl)
    spl = StaticPainterNDListener("XGBR_2d_Gasturbine_stat.png", "output", vars_indxs=[0, 1], mode="surface", calc="interpolation")
    solver.add_listener(spl)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()