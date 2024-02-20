from iOpt.method.search_data import SearchDataItem
from problems.grishagin_mco import Grishagin_mco
from iOpt.trial import Point
from iOpt.trial import FunctionValue
import numpy as np
from iOpt.method.multi_objective_optim_task import MultiObjectiveOptimizationTask
from iOpt.method.multi_objective_optim_task import MinMaxConvolution
from iOpt.method.optim_task import TypeOfCalculation

if __name__ == "__main__":
    """
    Минимизация тестовой функции Хилла c визуализацией
    """

    # создание объекта задачи
    problem = Grishagin_mco(2, [1, 2])
    # 1,30373711618 1,32060194666 часть решения, полученная прохоом по сетке
    # 0,70947330123 1,88734524329tel часть решения, полученная методом
    #problem = mco_test3()
    conv = MinMaxConvolution(problem, [0.5, 0.5], False)
    task = MultiObjectiveOptimizationTask(problem, conv)

    task.min_value = [np.double(-13.51436), np.double(-11.28447)]

    min_z = 100
    min_point = Point([], [])
    min_fun = np.ndarray(shape=(2,), dtype=FunctionValue)

    # проход по сетке 100Х100 и поиск точки с минимальным значением z
    for i in range(0, 100):
        for j in range(0, 100):
            pointfv = [np.double(i/100), np.double(j/100)]
            point = Point(pointfv, [])

            funV = np.ndarray(shape=(2,), dtype=FunctionValue)
            for k in range(0, 2):
                funV[k] = FunctionValue(functionID=k)
                funV[k] = problem.calculate(point, funV[k])

            sditem = SearchDataItem(point, point.float_variables[1], funV, [])
            task.calculate(sditem, -1, TypeOfCalculation.CONVOLUTION)

            if (min_z > sditem.get_z()):
                min_z = sditem.get_z()
                min_point = sditem.point
                min_fun = funV
            #funV = self.calculate(KOpoint, KOfunV[0])

    print(min_z, min_point.float_variables, [fv.value for fv in min_fun])
    # 1.3206019466549357 [0.63, 0.37] [-10.906885767640356, -8.64326610669013]

    #is_scaling
    # 0.11702826509839945 [0.63, 0.37] [-10.906885767640356, -8.64326610669013]

    ""
    # Область парето после 5000 итераций
    # [-10.20071994533217, -9.048618851776029][0.62353516 0.36083984]
    # [-8.301130179104787, -10.40551296979958][0.65185547 0.35107422]
    # [-13.144906394676253, -6.072324595007004][0.61181641 0.39404297]
    # [-11.938589250054529, -7.441058563760394][0.62939453 0.38427734]
    # [-10.190288362639677, -9.221841563370118][0.63330078 0.36376953]
    # [-7.204550076392398, -11.000241595429669][0.66455078 0.33349609]
    # [-5.7913570098004525, -11.24874919452789][0.64697266 0.32080078]
    # [-9.302230472348885, -9.553057450921768][0.64892578 0.36376953]
    # [-9.15234020775477, -9.781148328755988][0.62939453 0.35302734]
    # [-12.848034559388259, -6.54532948000002][0.61669922 0.39013672]
    # [-10.3094368604201, -8.62635751498344][0.64404297 0.37451172]
    # [-11.27774822721674, -8.30936389536164][0.62841797 0.37353516]
    # [-9.121203314934272, -9.96846279713267][0.64111328 0.35595703]
    # [-11.910727880612416, -7.706230147786212][0.62060547 0.37744141]
    # [-8.294820285320384, -10.41957178118964][0.63916016 0.34716797]
    # [-12.200336647760743, -7.069230751600034][0.60791016 0.37841797]
    # [-7.637515158932618, -10.77240652397774][0.65869141 0.34326172]
    # [-7.207698192143824, -10.855790821057813][0.67333984 0.32470703]
    # [-12.96704615527787, -6.2097075149626155][0.60595703 0.38916016]
    # [-6.764254636468001, -11.2176851692763][0.65869141 0.32666016]
    # [-13.397922583064096, -5.481333524951266][0.60693359 0.39990234]
    # [-6.8004941888681, -11.203703457526622][0.65478516 0.32958984]
    # [-13.432857223048867, -4.758964996890917][0.60888672 0.41162109]
    ""

    #0.603052, 0.408337,  # f(min1)=-13.51436
    #0.652988, 0.320592,  # f(min2)=-11.28447
    #1.000000, 0.000000,  # f(min3)=-13.20907
    #0.066182, 0.582587,  # f(min4)=-11.54117
    #0.904308, 0.872639,  # f(min5)=-9.969261



