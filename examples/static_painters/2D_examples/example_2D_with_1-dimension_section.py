from iOpt.problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener

if __name__ == "__main__":
    # create the problem 2D dimension
    problem = Rastrigin(2)

    # add solver parameters
    params = SolverParameters(r=3.5, eps=0.001)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl_0 = StaticPaintListener("rastrigin_2_3.5_0.001_static1D_0.png", "output", indx=0)
    apl_1 = StaticPaintListener("rastrigin_2_3.5_0.001_static1D_1.png", "output", indx=1)
    solver.AddListener(apl_0)
    solver.AddListener(apl_1)

    # solve the problem
    sol = solver.Solve()
