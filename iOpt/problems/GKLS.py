import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from iOpt.problems.GKLS_function.gkls_random import GKLSRandomGenerator
from iOpt.problems.GKLS_function.gkls_function import T_GKLS_Minima
from iOpt.problems.GKLS_function.gkls_function import T_GKLS_GlobalMinima
from iOpt.problems.GKLS_function.gkls_function import GKLSClass
from iOpt.problems.GKLS_function.gkls_function import GKLSFuncionType
from iOpt.problems.GKLS_function.gkls_function import GKLSParameters
from iOpt.problems.GKLS_function.gkls_function import GKLSFunction

import math




class GKLS(Problem):
    """Base class for optimization problems"""

    def __init__(self, dimension: int, 
                 functionNumber: int = 1):
        self.dimension = dimension
        self.numberOfFloatVariables = dimension
        self.numberOfDisreteVariables = 0
        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.floatVariableNames = np.ndarray(shape=(dimension), dtype=str)
        for i in range(self.dimension):
            self.floatVariableNames[i] = i

        self.lowerBoundOfFloatVariables = np.ndarray(shape=(dimension), dtype=np.double)
        self.lowerBoundOfFloatVariables.fill(-1)
        self.upperBoundOfFloatVariables = np.ndarray(shape=(dimension), dtype=np.double)
        self.upperBoundOfFloatVariables.fill(1)

        self.function = GKLSFunction()

        mMaxDimension = 50;
        mMinDimension = 2;
        mNumberOfConstraints = 0;
        mLeftBorder = -1.0;
        mRightBorder = 1.0;

        function_number = functionNumber
        global_dist = 0.9
        global_radius= 0.33
        num_minima= 10
        
        problem_class= GKLSClass.Simple
        function_class= GKLSFuncionType.TD

        self.function.GKLS_global_dist = global_dist;
        self.function.GKLS_global_radius = global_radius;
        self.function.GKLS_global_value = -1.0;
        self.function.NumberOfLocalMinima = num_minima;
        self.function.SetDimension(self.dimension);
        self.function.mFunctionType = function_class;


        self.function.SetFunctionClass(problem_class, self.dimension);


        if (self.function.GKLS_parameters_check() != GKLSFunction.GKLS_OK):
            return;

        self.function.SetFunctionNumber(function_number);


        #self.knownOptimum = np.ndarray(shape=(1), dtype=Trial)
        #
        #pointfv = np.ndarray(shape=(dimension), dtype=np.double)
        #pointfv.fill(0)
        #KOpoint = Point(pointfv, [])
        #KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        #KOfunV[0] = FunctionValue()
        #KOfunV[0].value = 0
        #self.knownOptimum[0] = Trial(KOpoint, KOfunV)

        



    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        """Compute selected function at given point."""
        sum: np.double = 0        

        functionValue.value = self.function.Calculate(point.floatVariables);
        return functionValue
