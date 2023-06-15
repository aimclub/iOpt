from problems.rastriginInt import RastriginInt
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticDisreteListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    # create the problem 2D dimension
    problem = RastriginInt(6, 4)

    # add solver parameters
    params = SolverParameters(r=2.1, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticDisreteListener("RastriginInt_6-4_2.1_0.01_1.png", mode='analysis')
    solver.AddListener(apl)
    apl = StaticDisreteListener("RastriginInt_6-4_2.1_0.01_2.png", mode='bestcombination', subvars=[1, 2])
    solver.AddListener(apl)
    cfol = ConsoleOutputListener(mode="full")
    solver.AddListener(cfol)

    # solve the problem
    sol = solver.Solve()
