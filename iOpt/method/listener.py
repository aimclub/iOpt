from iOpt.method.search_data import SearchData


# интерфейс в методе
class Listener:
    def BeforeMethodStart(self, searchData: SearchData) -> None:
        pass

    def OnEndIteration(self, searchData: SearchData) -> None:
        pass

    def OnMethodStop(self, searchData: SearchData) -> None:
        pass

    def OnRefrash(self, searchData: SearchData) -> None:
        pass


# реализацию вынести за метод!
class FunctionPainter:
    def __init__(self, searchData: SearchData) -> None:
        self.searchData = searchData

    def Paint(self) -> None:
        pass


# пример слушателя
class PaintListener(Listener):
    # нарисовать все точки испытаний
    def OnMethodStop(self, searchData: SearchData) -> None:
        fp = FunctionPainter(searchData)
        fp.Paint()
