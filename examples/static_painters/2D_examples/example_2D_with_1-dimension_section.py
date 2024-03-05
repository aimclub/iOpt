from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterListener

if __name__ == "__main__":
    # create the problem _2D dimension
    problem = Rastrigin(2)

    # add solver parameters
    params = SolverParameters(r=3.5, eps=0.001)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl_0 = StaticPainterListener("rastrigin_2_3.5_0.001_static1D_0.png", "output", indx=0)
    apl_1 = StaticPainterListener("rastrigin_2_3.5_0.001_static1D_1.png", "output", indx=1)
    solver.add_listener(apl_0)
    solver.add_listener(apl_1)

    # solve the problem
    sol = solver.solve()
