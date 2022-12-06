from typing import List, Optional

from iOpt.method.console.console_output import FunctionConsoleFullOutput
from iOpt.method.method import Method
from iOpt.method.painters.dynamic_painter import FunctionAnimationNDPainter, FunctionAnimationPainter
from iOpt.method.painters.static_painter import FunctionStaticNDPainter, FunctionStaticPainter
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution


class Listener:
    def BeforeMethodStart(self, method: Method) -> None:
        pass

    def OnEndIteration(self, searchData: List[SearchDataItem]) -> None:
        pass

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool) -> None:
        pass

    def OnRefrash(self, searchData: SearchData, solution: Solution, status: bool) -> None:
        pass


class ConsoleFullOutputListener(Listener):
    def __init__(self, mode: int) -> None:
        self.__fcfo: Optional[FunctionConsoleFullOutput] = None
        self.mode = mode

    def BeforeMethodStart(self, method: Method) -> None:
        self.__fcfo = FunctionConsoleFullOutput(method.task.problem, method.parameters)
        self.__fcfo.printInitInfo()

    def OnEndIteration(self, savedNewPoints: List[SearchDataItem]) -> None:
        if self.__fcfo is None:
            raise RuntimeError("Method has not started")
        self.__fcfo.printIterInfo(savedNewPoints, self.mode)

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool) -> None:
        if self.__fcfo is None:
            raise RuntimeError("Method has not started")
        self.__fcfo.printFinalResult(solution, status)


class StaticPaintListener(Listener):
    def __init__(self, pathForSaves: str, fileName: str, indx: Optional[int] = None,
                 isPointsAtBottom: bool = True, toPaintObjFunc: bool = True) -> None:
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameterInNDProblem = indx
        self.isPointsAtBottom = isPointsAtBottom
        self.toPaintObjFunc = toPaintObjFunc

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool) -> None:
        if self.parameterInNDProblem is None:
            raise RuntimeError("parameterInNDProblem is None")
        fp = FunctionStaticPainter(searchData, solution)
        fp.Paint(self.fileName, self.pathForSaves, self.isPointsAtBottom,
                 self.parameterInNDProblem, self.toPaintObjFunc)


class StaticNDPaintListener(Listener):
    def __init__(self, pathForSaves: str, fileName: str,
                 varsIndxs: List[int] = [0, 1], toPaintObjFunc: bool = True) -> None:
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.toPaintObjFunc = toPaintObjFunc

    def OnMethodStop(self, searchData: SearchData,
                     solution: Solution, status: bool) -> None:
        fp = FunctionStaticNDPainter(searchData, solution)
        fp.Paint(self.fileName, self.pathForSaves, self.parameters, self.toPaintObjFunc)


class AnimationPaintListener(Listener):
    def __init__(self, pathForSaves: str, fileName: str, isPointsAtBottom: bool = True,
                 toPaintObjFunc: bool = True) -> None:
        self.__fp: Optional[FunctionAnimationPainter] = None
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.isPointsAtBottom = isPointsAtBottom
        self.toPaintObjFunc = toPaintObjFunc

    def BeforeMethodStart(self, method: Method) -> None:
        self.__fp = FunctionAnimationPainter(method.task.problem, self.isPointsAtBottom)
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc()

    def OnEndIteration(self, savedNewPoints: List[SearchDataItem]) -> None:
        if self.__fp is None:
            raise RuntimeError("Method has not started")
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool) -> None:
        if self.__fp is None:
            raise RuntimeError("Method has not started")
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)


class AnimationNDPaintListener(Listener):
    def __init__(self, pathForSaves: str, fileName: str, varsIndxs: List[int], toPaintObjFunc: bool = True) -> None:
        self.__fp: Optional[FunctionAnimationNDPainter] = None
        self.fileName = fileName
        self.pathForSaves = pathForSaves
        self.parameters = varsIndxs
        self.toPaintObjFunc = toPaintObjFunc

    def BeforeMethodStart(self, method: Method) -> None:
        self.__fp = FunctionAnimationNDPainter(method.task.problem, self.parameters)

    def OnEndIteration(self, savedNewPoints: List[SearchDataItem]) -> None:
        if self.__fp is None:
            raise RuntimeError("Method has not started")
        self.__fp.PaintPoint(savedNewPoints)

    def OnMethodStop(self, searchData: SearchData, solution: Solution, status: bool) -> None:
        if self.__fp is None:
            raise RuntimeError("Method has not started")
        if self.toPaintObjFunc:
            self.__fp.PaintObjectiveFunc(solution)
        self.__fp.PaintOptimum(solution, self.fileName, self.pathForSaves)
