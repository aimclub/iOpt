from iOpt.method.listener import StaticNDPaintListener, AnimationNDPaintListener, ConsoleFullOutputListener
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
    method_params = SolverParameters(r=np.double(2.0), itersLimit=200)
    solver = Solver(problem, parameters=method_params)
    apl = AnimationNDPaintListener("svc2d_anim.png", "output", varsIndxs=[0, 1], toPaintObjFunc=False)
    solver.AddListener(apl)
    spl = StaticNDPaintListener("svc2d_stat.png", "output", varsIndxs=[0, 1], mode="surface", calc="interpolation")
    solver.AddListener(spl)
    cfol = ConsoleFullOutputListener(mode='full')
    solver.AddListener(cfol)
    solver_info = solver.Solve()