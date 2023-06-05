from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

from sklearn.datasets import load_breast_cancer
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._3D.Problem import SVC_3D
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
    kernel_type = {'kernel': ['rbf', 'sigmoid']}

    problem = SVC_3D.SVC_3D(x, y, regularization_value_bound, kernel_coefficient_bound, kernel_type)

    method_params = SolverParameters(r=np.double(3.0), itersLimit=100)
    solver = Solver(problem, parameters=method_params)

    cfol = ConsoleOutputListener(mode='full')
    solver.AddListener(cfol)

    solver_info = solver.Solve()
