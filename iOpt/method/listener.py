from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.method.painters.static_painter import FunctionStaticPainter, FunctionStaticNDPainter
from iOpt.method.painters.dynamic_painter import FunctionAnimationPainter, FunctionAnimationNDPainter
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
    def __init__(self, fileName: str, pathForSaves="", indx=None, isPointsAtBottom=False, toPaintObjFunc=True):
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameterInNDProblem = indx
        self.isPointsAtBottom = isPointsAtBottom
        self.toPaintObjFunc = toPaintObjFunc

    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution, status : bool):
        fp = FunctionStaticPainter(searchData, solution)
        fp.Paint(self.fileName, self.pathForSaves, self.isPointsAtBottom,
        self.parameterInNDProblem, self.toPaintObjFunc)

class StaticNDPaintListener(Listener):
    def __init__(self, fileName : str, pathForSaves="", varsIndxs=[0,1], toPaintObjFunc=True):
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.toPaintObjFunc = toPaintObjFunc

    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution, status : bool):
        fp = FunctionStaticNDPainter(searchData, solution)
        fp.Paint(self.fileName, self.pathForSaves, self.parameters, self.toPaintObjFunc)

class AnimationPaintListener(Listener):
    def __init__(self, fileName : str, pathForSaves="", isPointsAtBottom=False, toPaintObjFunc=True):
        self.__fp : FunctionAnimationPainter = None
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.isPointsAtBottom = isPointsAtBottom
        self.toPaintObjFunc = toPaintObjFunc

    def BeforeMethodStart(self, method : Method):
        self.__fp = FunctionAnimationPainter(method.task.problem, self.isPointsAtBottom)
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc()

    def OnEndIteration(self, savedNewPoints : SearchDataItem):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status : bool):
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)

class AnimationNDPaintListener(Listener):
    def __init__(self, fileName : str, pathForSaves="", varsIndxs=[0,1], toPaintObjFunc=True):
        self.__fp : FunctionAnimationNDPainter = None
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.toPaintObjFunc = toPaintObjFunc

    def BeforeMethodStart(self, method : Method):
        self.__fp = FunctionAnimationNDPainter(method.task.problem, self.parameters)

    def OnEndIteration(self, savedNewPoints : SearchDataItem):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status : bool):
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc(solution)
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)