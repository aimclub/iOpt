from examples.Machine_learning.XGBoostRegression._3D.Problems import XGB_3D
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.output_system.listeners.static_painters import StaticDiscreteListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from sklearn.utils import shuffle
import numpy as np
import csv

def gasturbine_Dataset():
    x = []
    y = []
    with open(r"../Datasets/no_predict.csv") as file:
        file_reader = csv.reader(file, delimiter=";")
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
    booster_type = {'booster': ['gblinear', 'gbtree', 'dart']}

    problem = XGB_3D.XGB_3D(X, Y, learning_rate_bound, gamma_bound, booster_type)
    method_params = SolverParameters(r=np.double(2.0), iters_limit=1000, number_of_parallel_points=16,
                                     evolvent_density=12)
    solver = Solver(problem, parameters=method_params)
    apl = StaticDiscreteListener("experiment1.png", mode='analysis')
    solver.add_listener(apl)
    apl = StaticDiscreteListener("experiment2.png", mode='bestcombination', calc='interpolation', mrkrs=4)
    solver.add_listener(apl)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()