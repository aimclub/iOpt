from problems.mco_test1 import mco_test1
from problems.mco_test3 import mco_test3
from problems.mco_test5 import mco_test5
from problems.mco_test6 import mco_test6
from problems.grishagin_mco import Grishagin_mco
from iOpt.trial import Point
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Минимизация тестовой функции Хилла c визуализацией
    """

    # создание объекта задачи
    problem = Grishagin_mco(2, [2, 3])

    # Формируем параметры решателя
    params = SolverParameters(r=2, eps=0.01, iters_limit=1600, number_of_lambdas=4)#, start_lambdas=[[0, 1]])  # , number_of_parallel_points=8) #, start_lambdas=[0.1, 0.9]

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Решение задачи
    sol = solver.solve()

    print("Grishagin_mco(2, [2, 3])")
    print("(r=2, eps=0.01, iters_limit=1600, number_of_lambdas=1, start_lambdas=[0, 1])")
    print("scaling false, new process")

    i=0
    x1 = [trial.point.float_variables[0] for trial in sol.best_trials]
    x2 = [trial.point.float_variables[1] for trial in sol.best_trials]
    print(x1)
    print(x2)
    #for k in range(problem.number_of_objectives):
    fv1 = [trial.function_values[0].value for trial in sol.best_trials]
    fv2 = [trial.function_values[1].value for trial in sol.best_trials]
    print(fv1)
    print(fv2)

    plt.plot(x1, x2, 'ro')
    plt.show()

    plt.plot(fv1, fv2, 'ro')
    plt.show()


    # for trial in sol.best_trials:
    #     fv = [trial.function_values[k].value for k in range(2)] #range(self.task.problem.number_of_objectives)
    #     print(fv, trial.point.float_variables)
    #     # print(i, trial.function_values[0].value, trial.function_values[1].value, trial.point.float_variables)
    #     i += 1

    # solver.change_lambdas_for_mco([0.4, 0.6], Point(float_variables=[0.2, 0.2], discrete_variables=None), 200)
    # print(solver.parameters.iters_limit)
    # sol2 = solver.solve()

    ##solver.do_global_iteration(500)
    #


    # solver.parameters.lambdas = [0.4, 0.6]
    #solver.parameters.iters_limit = 1300
    # for i in range (0, 10):
    #     solver.change_lambdas_for_mco([0+i/10, 1-i/10])
    #     solver.do_global_iteration(300)
    #
    #     solver.change_lambdas_for_mco([1-i/10, 0+i/10])
    #     solver.do_global_iteration(300)




    #### solver.change_lambdas_for_mco([0.4, 0.6], Point(float_variables=[0.2, 0.2], discrete_variables=None))
    # solver.do_global_iteration(100)
    #
    # solver.change_lambdas_for_mco([0.6, 0.4], Point(float_variables=[0.8, 0.8], discrete_variables=None))
    # solver.do_global_iteration(100)




    #
    # solver.change_lambdas_for_mco([0.7, 0.3])
    # solver.do_global_iteration(500)
    #
    # solver.change_lambdas_for_mco([0.3, 0.7])
    # solver.do_global_iteration(500)
    #
    # solver.change_lambdas_for_mco([0.2, 0.8])
    # solver.do_global_iteration(500)
    #
    # solver.change_lambdas_for_mco([0.8, 0.2])
    # solver.do_global_iteration(500)



    #sol2 = solver.solve()

    # # Формируем параметры решателя
    # params = SolverParameters(r=3, eps=0.01, iters_limit=1100, lambdas=[0.4, 0.6])  # , number_of_parallel_points=8)
    # # Создаем решатель
    # solver = Solver(problem=problem, parameters=params)
    # # Добавляем вывод результатов в консоль
    # cfol = ConsoleOutputListener(mode='full')
    # solver.add_listener(cfol)
    # # Решение задачи
    # sol2 = solver.solve()

    # 1500 iter scaling = false
    # [-10.20071994533217, -9.048618851776029][0.62353516 0.36083984]
    # [-13.34531671577386, -4.115499960055939][0.59619141 0.41748047]
    # [-8.301130179104787, -10.40551296979958][0.65185547 0.35107422]
    # [-13.144906394676253, -6.072324595007004][0.61181641 0.39404297]
    # [-7.383363155187936, -10.497366535044591][0.63134766 0.33935547]
    # [-5.794251895399704, -11.036052067578607][0.65771484 0.30517578]
    # [-12.586427435545735, -6.287993426756631][0.60009766 0.38427734]
    # [-11.938589250054529, -7.441058563760394][0.62939453 0.38427734]
    # [-10.190288362639677, -9.221841563370118][0.63330078 0.36376953]
    # [-7.204550076392398, -11.000241595429669][0.66455078 0.33349609]
    # [-5.7913570098004525, -11.24874919452789][0.64697266 0.32080078]
    # 1.5722983852208454[0.62353516 0.36083984] -10.20071994533217 - 9.048618851776029

    # 1500 iter scaling = true
    # [-11.447989606733351, -8.167143799279023][0.62451172 0.37353516]
    # [-13.456539178194587, -4.670347165685965][0.59716797 0.40771484]
    # [-7.803448462802699, -10.705307652579371][0.65576172 0.34521484]
    # [-7.901997778214576, -10.138273719954364][0.62646484 0.34326172]
    # [-5.989515579593246, -11.17448394308981][0.65771484 0.31103516]
    # [-9.918064533479935, -8.38741502981965][0.60791016 0.35791016]
    # [-13.01983494390576, -6.178971368531994][0.60693359 0.39013672]
    # [-9.562652011760534, -9.664441316894472][0.63916016 0.35986328]
    # [-7.220386202726583, -10.876942564653419][0.67138672 0.32958984]
    # [-6.0438904192101095, -11.109551622960772][0.64111328 0.32666016]
    # 0.13202730057681653 [0.60791016 0.35791016] -9.918064533479935 - 8.38741502981965
