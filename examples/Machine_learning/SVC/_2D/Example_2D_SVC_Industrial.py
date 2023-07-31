from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import SVC_2d
from sklearn.utils import shuffle
import numpy as np
import pandas as pd


def get_SCANIA_dataset():
    xls = pd.read_excel(r"../Datasets/aps_failure_training_set1.xls", header=None)
    data = xls.values[1:]
    row, col = data.shape
    _x = data[:, 1:col]
    _y = data[:, 0]
    y = np.array(_y, dtype=np.double)
    x = np.array(_x, dtype=np.double)
    return shuffle(x, y, random_state=42)


if __name__ == "__main__":
    X, Y = get_SCANIA_dataset()
    x = X[:2000]
    y = Y[:2000]
    regularization_value_bound = {'low': 1, 'up': 10}
    kernel_coefficient_bound = {'low': -8, 'up': -1}
    problem = SVC_2d.SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)
    method_params = SolverParameters(r=np.double(2.0), iters_limit=200)
    solver = Solver(problem, parameters=method_params)
    apl = AnimatePainterNDListener("svc2d_anim.png", "output", vars_indxs=[0, 1])
    solver.add_listener(apl)
    spl = StaticPainterNDListener("svc2d_stat.png", "output", vars_indxs=[0, 1], mode="surface", calc="interpolation")
    solver.add_listener(spl)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()
