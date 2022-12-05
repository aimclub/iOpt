from iOpt.trial import FunctionValue


class SolutionValue(FunctionValue):
    def __init__(self,
                 calculationsNumber: int = -1,
                 holderConstantsEstimations: float = -1.0
                 ):
        self.calculationsNumber = calculationsNumber
        self.holderConstantsEstimations = holderConstantsEstimations
