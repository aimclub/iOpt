import numpy as np
from iOpt.trial import FunctionValue


class SolutionValue(FunctionValue):
    def __init__(self,
                 calculations_number: int = -1,
                 holder_constants_estimations: np.double = -1.0
                 ):
        self.calculationsNumber = calculations_number
        self.holder_constants_estimations = holder_constants_estimations
