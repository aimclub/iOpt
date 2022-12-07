import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from sko.GA import GA_TSP
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict

class SVC_2D(Problem):

    def __init__(self, xdataset: np.ndarray, ydataset: np.ndarray,
                 regularizationbound: Dict[str, float],
                 kernelcoefficientbound: Dict[str, float]):
        self.dimension = 2
        self.numberOfFloatVariables = 2
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        if xdataset.shape[0] != ydataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = xdataset
        self.y = ydataset
        self.floatVariableNames = np.array(["Regularization parameter", "Kernel coefficient"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([regularizationbound['low'], kernelcoefficientbound['low']], dtype=np.double)
        self.upperBoundOfFloatVariables = np.array([regularizationbound['up'], kernelcoefficientbound['up']], dtype=np.double)

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        cs, gammas = point.floatVariables[0], point.floatVariables[1]
        clf = SVC(C=10**cs, gamma=10 ** gammas)
        clf.fit(self.x, self.y)
        functionValue.value = -cross_val_score(clf, self.x, self.y,
                                              scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()
        return functionValue
