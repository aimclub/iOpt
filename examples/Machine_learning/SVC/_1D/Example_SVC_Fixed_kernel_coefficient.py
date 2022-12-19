from iOpt.method.listener import StaticPaintListener, AnimationPaintListener
from sklearn.datasets import load_breast_cancer
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.SVC._1D.Problems import SVC_Fixed_Kernel
from sklearn.utils import shuffle
import numpy as np


def load_breast_cancer_data():
    dataset = load_breast_cancer()
    x_raw, y_raw = dataset['data'], dataset['target']
    inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
    return inputs, outputs


if __name__ == "__main__":
    x, y = load_breast_cancer_data()
    kernel_coefficient = -5
    regularization_value_bound = {'low': 1, 'up': 6}
    problem = SVC_Fixed_Kernel.SVC_Fixed_Kernel(x, y, kernel_coefficient, regularization_value_bound)

    method_params = SolverParameters(r=np.double(3.0), eps=np.double(0.05))
    solver = Solver(problem, parameters=method_params)

    apl = AnimationPaintListener("svc1d_anim.png", "output", toPaintObjFunc=True)
    solver.AddListener(apl)

    spl = StaticPaintListener("svc1d_stat.png", "output", mode="interpolation")
    solver.AddListener(spl)

    solver_info = solver.Solve()
    print(solver_info.numberOfGlobalTrials)
    print(solver_info.numberOfLocalTrials)
    print(solver_info.solvingTime)

    print(solver_info.bestTrials[0].point.floatVariables)
    print(solver_info.bestTrials[0].functionValues[0].value)
