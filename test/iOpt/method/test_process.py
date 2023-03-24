import unittest
from unittest import mock
from unittest.mock import Mock

import numpy as np

from iOpt.method.optim_task import TypeOfCalculation
from iOpt.trial import Point
from iOpt.trial import Trial
from iOpt.trial import FunctionValue
from iOpt.method.process import Process
from iOpt.method.method import Method
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters


class TestProcess(unittest.TestCase):
    @mock.patch('iOpt.problem.Problem')
    @mock.patch('iOpt.method.optim_task.OptimizationTask')
    @mock.patch('iOpt.evolvent.evolvent')
    @mock.patch('iOpt.method.listener.Listener')
    def setUp(self, mock_problem, mock_task, mock_evolvent, mock_listener):
        mock_problem.lowerBoundOfFloatVariables = [0.0]
        mock_problem.upperBoundOfFloatVariables = [1.0]
        searchData = SearchData(mock_problem)
        functionValue = FunctionValue()
        functionValue.value = -5.0
        searchData.solution.bestTrials = [Trial(Point([0.45], ['a']), [functionValue])]
        method = Method(evolvent=mock_evolvent, parameters=SolverParameters(),
                        task=mock_task, searchData=searchData)
        method.stop = True
        method.dimension = 1
        self.process = Process(parameters=SolverParameters(), task=mock_task, evolvent=mock_evolvent,
                               searchData=searchData, method=method, listeners=mock_listener)

    def test_SolveAssertException(self):
        self.process.parameters = None
        self.process.method.parameters = None
        with self.assertRaises(BaseException):
            self.process.Solve()

    def test_DoLocalRefinementItersLimit(self):
        self.process.task.problem.Calculate = Mock(side_effect=self.Calculate)
        self.process.parameters.itersLimit = 40

        self.process.DoLocalRefinement(-1)
        self.assertEqual(40 * 0.05, self.process.localMethodIterationCount)
        self.assertEqual(0.45, self.process.searchData.solution.bestTrials[0].point.floatVariables)
        self.assertEqual(-10.25, self.process.searchData.solution.bestTrials[0].functionValues[0].value)

    def CheckStopCondition(self) -> bool:
        # для выполнения в методе Solve одного вызова DoGlobalIteration()
        self.process.method.stop = not self.process.method.stop
        return self.process.method.stop

    def test_SolveRefineSolution(self):
        self.process.method.evolvent.GetImage = Mock(side_effect=self.GetImage)
        self.process.task.Calculate = Mock(side_effect=self.CalculateTask)
        self.process.task.problem.Calculate = Mock(side_effect=self.Calculate)
        self.process.method.CheckStopCondition = Mock(side_effect=self.CheckStopCondition)
        self.process.parameters.refineSolution = True
        try:
            self.process.Solve()
        except Exception:
            self.fail("...")

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        functionValue.value = (point.floatVariables[0] - 0.45)**2 - 10.25
        return functionValue

    def test_problemCalculate(self):
        self.process.task.problem.Calculate = Mock(side_effect=self.Calculate)
        self.assertEqual(-10.25, self.process.problemCalculate([0.45]))
        self.process.task.problem.Calculate.assert_called_once()

    def test_GetResult(self):
        self.assertEqual(1, len(self.process.searchData.solution.bestTrials))
        self.assertEqual(0.45, self.process.searchData.solution.bestTrials[0].point.floatVariables[0])
        self.assertEqual('a', self.process.searchData.solution.bestTrials[0].point.discreteVariables[0])
        self.assertEqual(-5.0, self.process.searchData.solution.bestTrials[0].functionValues[0].value)
        self.assertEqual(0.0, self.process.searchData.solution.problem.lowerBoundOfFloatVariables[0])
        self.assertEqual(1.0, self.process.searchData.solution.problem.upperBoundOfFloatVariables[0])


    def GetImage(self, x: np.double):
        return [x]

    def CalculateTask(self, dataItem: SearchDataItem, functionIndex: int = 0,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION) -> SearchDataItem:
        dataItem.functionValues[0] = self.Calculate(dataItem.point, dataItem.functionValues[0])
        return dataItem

    def test_DoGlobalIteration(self):
        self.process.method.evolvent.GetImage = Mock(side_effect=self.GetImage)
        self.process.task.Calculate = Mock(side_effect=self.CalculateTask)
        try:
            self.process.DoGlobalIteration(2)
        except Exception:
            self.fail("...")


if __name__ == '__main__':
    unittest.main()
