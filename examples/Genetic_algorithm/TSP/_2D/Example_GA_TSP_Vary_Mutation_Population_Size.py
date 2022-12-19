from iOpt.method.listener import StaticNDPaintListener, AnimationNDPaintListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Genetic_algorithm.TSP._2D.Problems import ga_tsp_2d
import numpy as np
import xml.etree.ElementTree as ET


def load_TSPs_matrix(filename):
    root = ET.parse(filename).getroot()
    columns = root.findall('graph/vertex')
    num_cols = len(columns)
    trans_matrix = np.zeros((num_cols, num_cols))
    for i, v in enumerate(columns):
        for e in v:
            j = int(e.text)
            trans_matrix[i, j] = float(e.get('cost'))
    return trans_matrix


if __name__ == "__main__":
    tsp_matrix = load_TSPs_matrix('../TSPs_matrices/a280.xml')
    num_iteration = 200
    mutation_probability_bound = {'low': 0.0, 'up': 1.0}
    population_size_bound = {'low': 10.0, 'up': 100.0}
    problem = ga_tsp_2d.GA_TSP_2D(tsp_matrix, num_iteration,
                                  mutation_probability_bound, population_size_bound)

    method_params = SolverParameters(r=np.double(2.0), itersLimit=300)
    solver = Solver(problem, parameters=method_params)

    apl = AnimationNDPaintListener("gatsp_2d_anim_vary_mutation.png", "output",  varsIndxs=[0, 1], toPaintObjFunc=True)
    solver.AddListener(apl)

    spl = StaticNDPaintListener("gatsp_2d_stat_vary_mutation.png", "output", varsIndxs=[0, 1], mode="interpolation")
    solver.AddListener(spl)

    solver_info = solver.Solve()
    print(solver_info.numberOfGlobalTrials)
    print(solver_info.numberOfLocalTrials)
    print(solver_info.solvingTime)

    print(solver_info.bestTrials[0].point.floatVariables)
    print(solver_info.bestTrials[0].functionValues[0].value)
