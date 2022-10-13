import numpy as np


class Evolvent:
    def __init__(self,
                 lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = [],
                 upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double) = [],
                 numberOfFloatVariables: int = 1,
                 evolventDensity: int = 10
                ):
        self.numberOfFloatVariables = numberOfFloatVariables
        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables
        self.evolventDensity = evolventDensity

    def GetImage(self,
                 x: np.double
                ) -> np.ndarray(shape = (1), dtype = np.double):
        """
        x->y
        """
        pass

    def GetInverseImage(self,
                        y: np.ndarray(shape = (1), dtype = np.double)
                       ) -> np.double:
        """
        y->x
        """
        pass

