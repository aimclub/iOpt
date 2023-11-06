from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.output_system.listeners.static_painters import StaticDiscreteListener
from sklearn.datasets import load_breast_cancer
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._3D.Problem import SVC_3D
from sklearn.utils import shuffle


def load_breast_cancer_data():
    dataset = load_breast_cancer()
    x_raw, y_raw = dataset['data'], dataset['target']
    inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
    return inputs, outputs

if __name__ == "__main__":
    x, y = load_breast_cancer_data()
    regularization_value_bound = {'low': 1, 'up': 10}
    kernel_coefficient_bound = {'low': -9, 'up': -6.7}
    kernel_type = {'kernel': ['rbf', 'sigmoid', 'poly']}
    problem = SVC_3D.SVC_3D(x, y, regularization_value_bound, kernel_coefficient_bound, kernel_type)
    method_params = SolverParameters(iters_limit=400)
    solver = Solver(problem, parameters=method_params)
    apl = StaticDiscreteListener("experiment1.png", mode='analysis')
    solver.add_listener(apl)
    apl = StaticDiscreteListener("experiment2.png", mode='bestcombination', calc='interpolation', mrkrs=4)
    solver.add_listener(apl)
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()

