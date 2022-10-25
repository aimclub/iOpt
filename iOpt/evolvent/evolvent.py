import numpy as np


class Evolvent:
    # конструктор класса
    # ------------------
    def __init__(self,
                 # массив для левых (нижних) границ, А
                 lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = [],
                 # массив для правых (верхних) границ, В
                 upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = [],
                 # N
                 numberOfFloatVariables: int = 1,
                 # m
                 evolventDensity: int = 10
                ):

        self.numberOfFloatVariables = numberOfFloatVariables
        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables
        self.evolventDensity = evolventDensity

        self.nexpValue = 0 # nexpExtended
        self.yValues: np.ndarray(shape = (1), dtype = np.double) = [0,0] # y

    # Установка границ
    # ----------------
    def SetBounds(self,
                 # массив для левых (нижних) границ, А
                 lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = [],
                 # массив для правых (верхних) границ, В
                 upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = []
                 ):
        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables    

    # Получить (x->y)
    # ---------------
    def GetImage(self,
                 x: np.double
                ) -> np.ndarray(shape = (1), dtype = np.double):

        self.__GetYonX(x)
        self.__TransformP2D()
        return self.yValues

    # Получить (y->x)
    # ----------------
    def GetInverseImage(self,
                        y: np.ndarray(shape = (1), dtype = np.double)
                       ) -> np.double:

        self.yValues = y
        self.__TransformD2P()
        x = self.__GetXonY()
        return x
    
    # ----------------------
    def GetPreimages(self,
                 y: np.ndarray(shape = (1), dtype = np.double),
                ) -> np.double:
        self.yValues = y
        self.__TransformD2P()
        x = self.__GetXonY()
        return x

    # Преобразование 
    # --------------------------------
    def  __TransformP2D(self):
        for i in range(0, self.numberOfFloatVariables):
            self.yValues[i] = self.yValues[i] * (self.upperBoundOfFloatVariables[i] - self.lowerBoundOfFloatVariables[i]) +\
            (self.upperBoundOfFloatVariables[i] + self.lowerBoundOfFloatVariables[i]) / 2
    
    # Преобразование
    # --------------------------------
    def  __TransformD2P(self):
        for i in range(0, self.numberOfFloatVariables):
            self.yValues[i] = (self.yValues[i] - (self.upperBoundOfFloatVariables[i] + self.lowerBoundOfFloatVariables[i]) /2) /\
            (self.upperBoundOfFloatVariables[i] - self.lowerBoundOfFloatVariables[i]) 
    
    #---------------------------------
    def __GetYonX(self, _x: np.double) -> np.ndarray(shape = (1), dtype = np.double):
        if self.numberOfFloatVariables == 1:
            self.yValues[0] = _x - 0.5
            return self.yValues

    #---------------------------------
    def __GetXonY(self) -> np.double:
        x: np.double
        if self.numberOfFloatVariables == 1:
            x = self.yValues[0] + 0.5 
            return x



   