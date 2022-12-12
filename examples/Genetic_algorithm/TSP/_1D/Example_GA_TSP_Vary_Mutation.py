from iOpt.method.listener import StaticPaintListener, AnimationPaintListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Genetic_algorithm.TSP._1D.Problems import ga_tsp_vary_mutation
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
    population_size = 100
    mutation_probability_bound = {'low': 0.0, 'up': 1.0}
    problem = ga_tsp_vary_mutation.GA_TSP_Vary_Mutation(tsp_matrix, num_iteration,
                                                        population_size, mutation_probability_bound)

    method_params = SolverParameters(r=np.double(3.0), itersLimit=100)
    solver = Solver(problem, parameters=method_params)


    apl = AnimationPaintListener("gatsp_1d_anim_vary_mutation.png", "output", toPaintObjFunc=True)
    solver.AddListener(apl)

    spl = StaticPaintListener("gatsp_1d_stat_vary_mutation.png", "output", mode="interpolation")
    solver.AddListener(spl)


    solver_info = solver.Solve()
    print(solver_info.numberOfGlobalTrials)
    print(solver_info.numberOfLocalTrials)
    print(solver_info.solvingTime)

    print(solver_info.bestTrials[0].point.floatVariables)
    print(solver_info.bestTrials[0].functionValues[0].value)