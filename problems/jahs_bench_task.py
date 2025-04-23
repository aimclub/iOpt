import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
import jahs_bench


class jahs_bench_task(Problem):
    """
    Class for solving a problem from juhs_bench_201 framework
    Parameters for optimization:
    - Learning rate (float variable, range: [0.001, 1.0])
    - Weight Decay (float variable, range: [0.00001, 0.01])
    - Kernel/activation layer (discrete variable, values: [ReLU, Mish, Hardswish])
    Dataset: cifar10
    Metric for optimization: accuracy
    """

    def __init__(self):
        """
        Constructor for the juhs_bench task
        """
        super(jahs_bench_task, self).__init__()
        self.name = "jahs_bench_task"
        self.dimension = 3
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.float_variable_names = np.array(["LearningRate", "WeightDecay"], dtype=str)
        self.lower_bound_of_float_variables = np.array([0.001, 0.00001], dtype=np.double)
        self.upper_bound_of_float_variables = np.array([1.0,0.01], dtype=np.double)
        self.discrete_variable_names.append('kernel')
        self.discrete_variable_values.append(['ReLU', 'Mish', 'Hardswish'])

        self.benchmark = jahs_bench.Benchmark(task="cifar10", kind="surrogate", 
          download=True, metrics=['valid-acc'])

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point
        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        p1 = point.float_variables[0]
        p2 = point.float_variables[1]
        kernel_type = point.discrete_variables[0]

        config = {
            'Optimizer': 'SGD',
            'LearningRate': p1,
            'WeightDecay': p2,
            'Activation': kernel_type,
            'TrivialAugment': False,
            'Op1': 4,
            'Op2': 1,
            'Op3': 2,
            'Op4': 0,
            'Op5': 2,
            'Op6': 1,
            'N': 5,
            'W': 16,
            'Resolution': 1.0,
        }

        results = self.benchmark(config, nepochs=200)
        val = results[200]['valid-acc']
        function_value.value = 100.0 - val

        return function_value
