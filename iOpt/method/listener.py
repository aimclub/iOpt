from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method

from iOpt.output_system.painters.dynamic_painter import FunctionAnimationPainter, FunctionAnimationNDPainter
from iOpt.output_system.console.console_output import FunctionConsoleFullOutput

class Listener:
    def BeforeMethodStart(self, searchData: SearchData):
        pass

    def OnEndIteration(self, searchData: SearchData, solution: Solution):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass

class ConsoleFullOutputListener(Listener):
    def __init__(self, mode = "static full", iters = 100):
        self.__fcfo : FunctionConsoleFullOutput = None
        self.mode = mode
        self.iters = iters

    def BeforeMethodStart(self, method : Method):
        self.__fcfo = FunctionConsoleFullOutput(method.task.problem, method.parameters)
        self.__fcfo.printInitInfo()
        pass

    def OnEndIteration(self, savedNewPoints : SearchDataItem, solution: Solution):
        if self.mode == "full":
            self.__fcfo.printIterPointInfo(savedNewPoints)
        elif self.mode == "custom":
            self.__fcfo.printBestPointInfo(solution, self.iters)
        elif self.mode == "result":
            pass

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status: bool):
        self.__fcfo.printFinalResult(solution, status)
        pass

# moode: objective function, approximation, only points
class StaticPaintListener(Listener):
    def __init__(self, fileName: str, pathForSaves="", indx=0, isPointsAtBottom=False, mode="objective function"):
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameterInNDProblem = indx
        self.isPointsAtBottom = isPointsAtBottom
        self.mode = mode

    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution, status : bool):
        fp = FunctionStaticPainter(searchData, solution)
        if self.mode=="objective function":
            fp.Paint(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem, True)
        elif self.mode=="only points":
            fp.Paint(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem, False)
        elif self.mode == "approximation":
            fp.PaintApproximation(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem)
        elif self.mode == "interpolation":
            fp.PaintInterpolation(self.fileName, self.pathForSaves, self.isPointsAtBottom, self.parameterInNDProblem)

# mode: surface, lines layers, approximation
class StaticNDPaintListener(Listener):
    def __init__(self, fileName : str, pathForSaves="", varsIndxs=[0,1], mode="lines layers", calc="objective function"):
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.mode = mode
        self.calc = calc
    def OnMethodStop(self, searchData: SearchData,
                    solution: Solution, status : bool, ):
        fp = FunctionStaticNDPainter(searchData, solution)
        if self.mode == "lines layers":
            if self.calc == "objective function":
                fp.PaintLL(self.fileName, self.pathForSaves, self.parameters)
            elif self.calc == "interpolation":
                fp.PaintLLI(self.fileName, self.pathForSaves, self.parameters)
        elif self.mode == "approximation":
            fp.PaintApproximation(self.fileName, self.pathForSaves, self.parameters)
        elif self.mode == "interpolation":
            fp.PaintInterpolation(self.fileName, self.pathForSaves, self.parameters)
        '''
        elif self.mode == "surface": # нужен ли?
            fp.PaintSurface(self.fileName, self.pathForSaves, self.parameters)
        '''
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

    def OnEndIteration(self, savedNewPoints : SearchDataItem, solution: Solution):
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

    def OnEndIteration(self, savedNewPoints : SearchDataItem, solution: Solution):
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData : SearchData, solution: Solution, status : bool):
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc(solution)
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)