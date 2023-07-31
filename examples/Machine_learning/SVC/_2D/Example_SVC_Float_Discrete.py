from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from sklearn.datasets import load_breast_cancer
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._2D.Problems import SVC_2D_Float_Discrete
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
    kernel_type = {'kernel': ['linear', 'rbf', 'sigmoid']}

    problem = SVC_2D_Float_Discrete.SVC_2D_Float_Discrete(x, y, regularization_value_bound, kernel_type)

    method_params = SolverParameters(r=np.double(3.0), iters_limit=100)
    solver = Solver(problem, parameters=method_params)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    solver_info = solver.solve()
