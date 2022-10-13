from search_data import SearchData


class Listener:
    def BeforeMethodStart(self, searchData: SearchData):
        pass

    def OnEndIteration(self, searchData: SearchData):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass


class FunctionPainter:
    def __init__(self, searchData: SearchData):
        self.searchData = searchData

    def Paint(self):
        pass


# пример слушателя
class PaintListener(Listener):
    # нарисовать все точки испытаний
    def OnMethodStop(self, searchData: SearchData):
        fp = FunctionPainter(searchData)
        fp.Paint()
        pass
