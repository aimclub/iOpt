from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.output_system.painters.dynamic_painter import FunctionAnimationPainter, FunctionAnimationNDPainter
from iOpt.output_system.painters.static_painter import FunctionStaticNDPainter, FunctionStaticPainter
from iOpt.output_system.console.console_output import FunctionConsoleFullOutput


class Listener:
    """
    Базовый класс слушателя событий.
    """

    def BeforeMethodStart(self, searchData: SearchData):
        pass

    def OnEndIteration(self, searchData: SearchData, solution: Solution):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass


class ConsoleFullOutputListener(Listener):
    """
    Класс ConsoleFullOutputListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на событие консольный вывод.
    """

    def __init__(self, mode='full', iters=100):
        """
        Конструктор класса ConsoleFullOutputListener

        :param mode: Режим вывода в консоль, который будет использован. Возможные режимы: 'full', 'custom' и 'result'.
           Режим 'full' осуществляет в процессе оптимизации полный вывод в консоль получаемой методом поисковой
           информации. Режим 'custom' осуществляет вывод текущей лучшей точки с заданной частотой. Режим 'result'
           выводит в консоль только финальный результат процесса оптимизации.
        :param iters: Частота вывода в консоль. Используется совместно с режимом вывода 'custom'.
        """
        self.__fcfo: FunctionConsoleFullOutput = None
        self.mode = mode
        self.iters = iters

    def BeforeMethodStart(self, method: Method):
        self.__fcfo = FunctionConsoleFullOutput(method.task.problem, method.parameters)
        self.__fcfo.printInitInfo()
        pass

    def OnEndIteration(self, savedNewPoints: SearchDataItem, solution: Solution):
        if self.mode == 'full':
            self.__fcfo.printIterPointInfo(savedNewPoints)
        elif self.mode == 'custom':
            self.__fcfo.printBestPointInfo(solution, self.iters)
        elif self.mode == 'result':
            pass

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool):
        self.__fcfo.printFinalResult(solution, status)
        pass


# moode: objective function, approximation, only points
class StaticPaintListener(Listener):
    """
    Класс StaticPaintListener - слушатель событий. Содержит метод-обработчик, выдающий в качестве
      реакции на завершение работы метода изображение.
    """

    def __init__(self, fileName: str, pathForSaves="", indx=0, isPointsAtBottom=False, mode='objective function'):
        """
        Конструктор класса StaticPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param indx: Индекс переменной оптимизационной задачи. Используется в многомерной оптимизации.
           Позволяет отобразить в сечении найденного минимума процесс оптимизации по одной выбранной переменной.
        :param isPointsAtBottom: Должны ли точки поисковой информации ставиться под графиком или нет. Если False,
           точки ставятся на графике.
        :param mode: Способ вычислений для отрисовки графика целевой функции, который будет использован. Возможные
           режимы: 'objective function', 'only points', 'approximation' и 'interpolation'. Режим 'objective function'
           строит график, вычисляя значения целевой функции на равномерной сетке. Режим 'approximation' строит
           нейроаппроксимацию для целевой функции на основе полученной поисковой информации.
           Режим 'interpolation' строит интерполяцию для целевой функции на основе полученной поисковой информации.
           Режим 'only points' не строит график целевой функции.
        """
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameterInNDProblem = indx
        self.isPointsAtBottom = isPointsAtBottom
        self.mode = mode

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool):
        fp = FunctionStaticPainter(searchData, solution)
        if self.mode == 'objective function':
            fp.Paint(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem, True)
        elif self.mode == 'only points':
            fp.Paint(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem, False)
        elif self.mode == 'approximation':
            fp.PaintApproximation(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem)
        elif self.mode == 'interpolation':
            fp.PaintInterpolation(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem)


# mode: surface, lines layers, approximation
class StaticNDPaintListener(Listener):
    """
    Класс StaticNDPaintListener - слушатель событий. Содержит метод-обработчик, выдающий в качестве
      реакции на завершение работы метода изображение.
      Используется для многомерной оптимизации.
    """

    def __init__(self, fileName: str, pathForSaves="", varsIndxs=[0, 1], mode='lines layers',
                 calc='objective function'):
        """
        Конструктор класса StaticNDPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param varsIndxs: Пара индексов переменных оптимизационной задачи, для которых будет построен рисунок.
        :param mode_: Режим отрисовки графика целевой функции, который будет использован.
           Возможные режимы:'lines layers', 'surface'.
           Режим 'lines layers' рисует линии уровня в сечении найденного методом решения.
           Режим 'surface' строит поверхность в сечении найденного методом решения.
        :param calc: Способ вычислений для отрисовки графика целевой функции, который будет использован. Возможные
           режимы: 'objective function' (только в режиме 'lines layers'), 'approximation' (только в режиме 'surface')
           и 'interpolation'. Режим 'objective function' строит график, вычисляя значения целевой функции на равномерной
           сетке. Режим 'approximation' строит нейроаппроксимацию для целевой функции на основе полученной поисковой
           информации. Режим 'interpolation' строит интерполяцию для целевой функции на основе полученной поисковой
           информации.
        """
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.mode = mode
        self.calc = calc

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool, ):
        fp = FunctionStaticNDPainter(searchData, solution)
        if self.mode == 'lines layers':
            if self.calc == 'objective function':
                fp.PaintLL(self.fileName, self.pathForSaves, self.parameters)
            elif self.calc == 'interpolation':
                fp.PaintLLI(self.fileName, self.pathForSaves, self.parameters)
            # elif self.calc == "approximation":
            #    pass # нужен ли?
        elif self.mode == 'surface':
            if self.calc == 'approximation':
                fp.PaintApproximation(self.fileName, self.pathForSaves, self.parameters)
            elif self.calc == 'interpolation':
                fp.PaintInterpolation(self.fileName, self.pathForSaves, self.parameters)
            # elif self.calc == "objective function":
            #    pass # нужен ли?


class AnimationPaintListener(Listener):
    """
    Класс AnimationPaintListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на события динамически обновляющееся изображение процесса оптимизации.
      Используется для одномерной оптимизации.
    """

    def __init__(self, fileName: str, pathForSaves="", isPointsAtBottom=False, toPaintObjFunc=True):
        """
        Конструктор класса AnimationPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param isPointsAtBottom: Должны ли точки поисковой информации ставиться под графиком или нет. Если False,
           точки ставятся на графике.
        :param toPaintObjFunc: Должна ли отрисовываться целевая функция или нет.
        """
        self.__fp: FunctionAnimationPainter = None
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.isPointsAtBottom = isPointsAtBottom
        self.toPaintObjFunc = toPaintObjFunc

    def BeforeMethodStart(self, method: Method):
        self.__fp = FunctionAnimationPainter(method.task.problem, self.isPointsAtBottom)
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc()

    def OnEndIteration(self, savedNewPoints: SearchDataItem, solution: Solution):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool):
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)


class AnimationNDPaintListener(Listener):
    """
    Класс AnimationPaintListener - слушатель событий. Содержит методы-обработчики, выдающие в качестве
      реакции на события динамически обновляющееся изображение процесса оптимизации.
      Используется для многомерной оптимизации.
    """

    def __init__(self, fileName: str, pathForSaves="", varsIndxs=[0, 1], toPaintObjFunc=True):
        """
        Конструктор класса AnimationNDPaintListener

        :param fileName: Название файла с указанием формата для сохранения изображения. Обязательный параметр.
        :param pathForSaves: Директория для сохранения изображения. В случае, если параметр не указан, изображение
           сохраняется в текущей рабочей директории.
        :param varsIndxs: Пара индексов переменных оптимизационной задачи, для которых будет построен рисунок.
        :param toPaintObjFunc: Должна ли отрисовываться целевая функция или нет.
        """
        self.__fp: FunctionAnimationNDPainter = None
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.toPaintObjFunc = toPaintObjFunc

    def BeforeMethodStart(self, method: Method):
        self.__fp = FunctionAnimationNDPainter(method.task.problem, self.parameters)

    def OnEndIteration(self, savedNewPoints: SearchDataItem, solution: Solution):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool):
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc(solution)
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)
