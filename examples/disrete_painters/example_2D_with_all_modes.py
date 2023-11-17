from problems.rastriginInt import RastriginInt
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticDiscreteListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    # create the problem _2D dimension float vars
    problem = RastriginInt(5, 3)

    # add solver parameters
    params = SolverParameters(r=2.1, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticDiscreteListener("RastriginInt_5-3_2.1_0.01_1.png", mode='analysis')
    solver.add_listener(apl)
    apl = StaticDiscreteListener("RastriginInt_5-3_2.1_0.01_2.png", mode='bestcombination', calc='objective function')
    solver.add_listener(apl)
    apl = StaticDiscreteListener("RastriginInt_5-3_2.1_0.01_3.png", mode='bestcombination', calc='interpolation')
    solver.add_listener(apl)
    cfol = ConsoleOutputListener(mode="full")
    solver.add_listener(cfol)

    solver.solve()
