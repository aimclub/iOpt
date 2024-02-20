from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from sklearn.datasets import load_breast_cancer
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import SVC_2d
from sklearn.utils import shuffle
import numpy as np


def load_breast_cancer_data():
    dataset = load_breast_cancer()
    x_raw, y_raw = dataset['data'], dataset['target']
    inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
    return inputs, outputs


if __name__ == "__main__":
    x, y = load_breast_cancer_data()
    regularization_value_bound = {'low': 1, 'up': 6}
    kernel_coefficient_bound = {'low': -7, 'up': -3}
    problem = SVC_2d.SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)
    method_params = SolverParameters(r=np.double(3.0), iters_limit=100)
    solver = Solver(problem, parameters=method_params)
    apl = AnimatePainterNDListener("svc2d_anim.png", "output", vars_indxs=[0, 1], to_paint_obj_func=False)
    solver.add_listener(apl)
    spl = StaticPainterNDListener("svc2d_stat.png", "output", vars_indxs=[0, 1], mode="surface", calc="interpolation")
    solver.add_listener(spl)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()
