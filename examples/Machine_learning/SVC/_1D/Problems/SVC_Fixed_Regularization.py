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

class SVC_Fixed_Regularization(Problem):

    def __init__(self, xdataset: np.ndarray, ydataset: np.ndarray, regularizationvalue: float,
                 kernelcoefficientBound: Dict[str, float]):
        self.dimension = 1
        self.numberOfFloatVariables = 1
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0
        if xdataset.shape[0] != ydataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = xdataset
        self.y = ydataset
        self.regularizationValue = regularizationvalue
        self.floatVariableNames = np.array(["Kernel coefficient"], dtype=str)
        self.lowerBoundOfFloatVariables = np.array([kernelcoefficientBound['low']], dtype=np.double)
        self.upperBoundOfFloatVariables = np.array([kernelcoefficientBound['up']], dtype=np.double)
        s = 0

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        kernel_coefficient = point.floatVariables[0]
        clf = SVC(C=10 ** self.regularizationValue, gamma=10 ** kernel_coefficient)
        clf.fit(self.x, self.y)
        functionValue.value = -cross_val_score(clf, self.x, self.y,
                                              scoring=lambda model, x, y: f1_score(y, model.predict(x))).mean()
        return functionValue