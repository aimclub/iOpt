from iOpt.method.listener import StaticPaintListener, AnimationPaintListener, StaticNDPaintListener, AnimationNDPaintListener
from sklearn.datasets import load_breast_cancer
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.problems.Machine_learning.SVC import SVC_2d
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

    method_params = SolverParameters(r=np.double(3.0), itersLimit=100)
    solver = Solver(problem, parameters=method_params)

    apl = AnimationNDPaintListener("output", "svc2d_anim.png", varsIndxs=[0, 1], toPaintObjFunc=True)
    solver.AddListener(apl)

    spl = StaticNDPaintListener("output", "svc2d_stat.png", varsIndxs=[0, 1], toPaintObjFunc=True)
    solver.AddListener(spl)

    solver_info = solver.Solve()
    print(solver_info.numberOfGlobalTrials)
    print(solver_info.numberOfLocalTrials)
    print(solver_info.solvingTime)

    print(solver_info.bestTrials[0].point.floatVariables)
    print(solver_info.bestTrials[0].functionValues[0].value)
