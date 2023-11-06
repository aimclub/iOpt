from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Genetic_algorithm.TSP._2D.Problems import ga_tsp_2d
import numpy as np
import xml.etree.ElementTree as ET


def load_TSPs_matrix(file_name):
    root = ET.parse(file_name).getroot()
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

    method_params = SolverParameters(r=np.double(2.0), iters_limit=300)
    solver = Solver(problem, parameters=method_params)

    apl = AnimatePainterNDListener("gatsp_2d_anim_vary_mutation.png", "output", vars_indxs=[0, 1],
                                   to_paint_obj_func=True)
    solver.add_listener(apl)

    spl = StaticPainterNDListener("gatsp_2d_stat_vary_mutation.png", "output", vars_indxs=[0, 1], mode="interpolation")
    solver.add_listener(spl)

    solver_info = solver.solve()
    print(solver_info.number_of_global_trials)
    print(solver_info.number_of_local_trials)
    print(solver_info.solving_time)

    print(solver_info.best_trials[0].point.float_variables)
    print(solver_info.best_trials[0].function_values[0].value)
