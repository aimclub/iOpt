from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.method.painters.static_painter import FunctionStaticPainter
from iOpt.method.painters.dynamic_painter import FunctionAnimationPainter
from iOpt.method.console.console_output import FunctionConsoleFullOutput

class Listener:
    def BeforeMethodStart(self, searchData: SearchData):
        pass

    def OnEndIteration(self, searchData: SearchData):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass

class ConsoleFullOutputListener(Listener):
    def __init__(self, mode):
        self.__fcfo : FunctionConsoleFullOutput = None
        self.mode = mode

    def BeforeMethodStart(self, method : Method):
        self.__fcfo = FunctionConsoleFullOutput(method.task.problem, method.parameters)
        self.__fcfo.printInitInfo()
        pass

    def OnEndIteration(self, savedNewPoints : SearchDataItem):
        self.__fcfo.printIterInfo(savedNewPoints, self.mode)
        pass

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status: bool):
        self.__fcfo.printFinalResult(solution, status)
        pass

class StaticPaintListener(Listener):
    def __init__(self, fileName):
        self.fileName = fileName

    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution, status : bool):
        fp = FunctionStaticPainter(searchData, solution)
        fp.Paint(self.fileName)

class AnimationPaintListener(Listener):
    def __init__(self, fileName):
        self.__fp : FunctionAnimationPainter = None
        self.fileName = fileName

    def BeforeMethodStart(self, method : Method):
        self.__fp = FunctionAnimationPainter(method.task.problem)
        self.__fp.PaintObjectiveFunc()

    def OnEndIteration(self, savedNewPoints : SearchDataItem):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status : bool):
        self.__fp.PaintOptimum(solution, self.fileName)
