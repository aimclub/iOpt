import unittest
from unittest import mock
from unittest.mock import Mock, call

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
from iOpt.method.listener import StaticPaintListener


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
        listener = StaticPaintListener(fileName="Output.txt")
        self.process = Process(parameters=SolverParameters(), task=mock_task, evolvent=mock_evolvent,
                               searchData=searchData, method=method, listeners=[listener])

    def test_SolveAssertException(self):
        self.process.parameters = None
        self.process.method.parameters = None
        with self.assertRaises(BaseException):
            self.process.Solve()

    def test_DoLocalRefinementItersLimit(self):
        self.process.task.problem.Calculate = Mock(side_effect=self.mock_Calculate)
        self.process.parameters.itersLimit = 40
        try:
            self.process.DoLocalRefinement(-1)
            self.assertEqual(4, self.process.searchData.solution.numberOfLocalTrials)
            self.assertEqual(40 * 0.05, self.process.localMethodIterationCount)
            self.assertEqual(0.45, self.process.searchData.solution.bestTrials[0].point.floatVariables)
            self.assertEqual(-10.25, self.process.searchData.solution.bestTrials[0].functionValues[0].value)
            self.process.task.problem.Calculate.assert_called()
        except Exception:
            self.fail("test_DoLocalRefinementItersLimit is failed")

    def mock_CheckStopCondition(self) -> bool:
        # для выполнения в методе Solve одного вызова DoGlobalIteration()
        self.process.method.stop = not self.process.method.stop
        return self.process.method.stop

    @mock.patch('iOpt.method.listener.StaticPaintListener.OnMethodStop')
    def test_SolveRefineSolutionAndCallListener(self, mock_OnMethodStop):
        self.process.method.evolvent.GetImage = Mock(side_effect=self.mock_GetImage)
        self.process.task.Calculate = Mock(side_effect=self.mock_CalculateTask)
        self.process.task.problem.Calculate = Mock(side_effect=self.mock_Calculate)
        self.process.method.CheckStopCondition = Mock(side_effect=self.mock_CheckStopCondition)
        self.process.parameters.refineSolution = True
        self.process.parameters.itersLimit = 20  # localMethodIterationCount = 20 * 0.05 = 2
        try:
            self.process.Solve()
            mock_OnMethodStop.assert_called_once()
            self.assertEqual(1, self.process.searchData.solution.numberOfGlobalTrials)
            self.assertEqual(2, self.process.searchData.solution.numberOfLocalTrials)
            self.process.method.evolvent.GetImage.assert_called()
            self.process.task.Calculate.assert_called()
            self.process.task.problem.Calculate.assert_called()
            self.process.method.CheckStopCondition.assert_called()

        except Exception:
            self.fail("test_SolveRefineSolution is failed")

    def mock_Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        functionValue.value = (point.floatVariables[0] - 0.45)**2 - 10.25
        return functionValue

    def test_problemCalculate(self):
        self.process.task.problem.Calculate = Mock(side_effect=self.mock_Calculate)
        self.assertEqual(-10.25, self.process.problemCalculate([0.45]))
        self.process.task.problem.Calculate.assert_called_once()

    def test_GetResult(self):
        self.assertEqual(1, len(self.process.searchData.solution.bestTrials))
        self.assertEqual(0.45, self.process.searchData.solution.bestTrials[0].point.floatVariables[0])
        self.assertEqual('a', self.process.searchData.solution.bestTrials[0].point.discreteVariables[0])
        self.assertEqual(-5.0, self.process.searchData.solution.bestTrials[0].functionValues[0].value)
        self.assertEqual(0.0, self.process.searchData.solution.problem.lowerBoundOfFloatVariables[0])
        self.assertEqual(1.0, self.process.searchData.solution.problem.upperBoundOfFloatVariables[0])

    def mock_GetImage(self, x: np.double):
        return [x]

    def mock_CalculateTask(self, dataItem: SearchDataItem, functionIndex: int = 0,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION) -> SearchDataItem:
        dataItem.functionValues[0] = self.mock_Calculate(dataItem.point, dataItem.functionValues[0])
        return dataItem

    @mock.patch('iOpt.method.listener.StaticPaintListener.BeforeMethodStart')
    @mock.patch('iOpt.method.listener.StaticPaintListener.OnEndIteration')
    def test_DoGlobalIterationAndListener(self, mock_OnEndIteration, mock_BeforeMethodStart):
        self.process.method.evolvent.GetImage = Mock(side_effect=self.mock_GetImage)
        self.process.task.Calculate = Mock(side_effect=self.mock_CalculateTask)
        try:
            self.process.DoGlobalIteration(1)
            mock_BeforeMethodStart.assert_called_once()
            mock_OnEndIteration.assert_called_once()
            self.assertEqual(1, self.process.searchData.solution.numberOfGlobalTrials)
            self.process.method.evolvent.GetImage.assert_called()
            self.process.task.Calculate.assert_called()
        except Exception:
            self.fail("test_DoGlobalIteration is failed")

    @mock.patch('iOpt.method.method.Method.FirstIteration')
    @mock.patch('iOpt.method.search_data.SearchData.GetLastItem')
    @mock.patch('iOpt.method.method.Method.CalculateIterationPoint',
                return_value=[SearchDataItem(Point(0.25, None), 0.25), SearchDataItem(Point(0.5, None), 0.5)])
    @mock.patch('iOpt.method.method.Method.CalculateFunctionals',
                return_value=SearchDataItem(Point(0.25, None), 0.25, -0.45))
    @mock.patch('iOpt.method.method.Method.UpdateOptimum')
    @mock.patch('iOpt.method.method.Method.RenewSearchData')
    def test_DoGlobalIterationFirst(self, mock_RenewSearchData, mock_UpdateOptimum,
                                    mock_CalculateFunctionals, mock_CalculateIterationPoint,
                                    mock_GetLastItem, mock_FirstIteration):
        self.process.DoGlobalIteration(2)
        expected_calls = [call().method6()]
        mock_FirstIteration.assert_has_calls(expected_calls, any_order=False)
        mock_CalculateIterationPoint.assert_has_calls([call().method4()], any_order=False)
        mock_FirstIteration.assert_called_once()


if __name__ == '__main__':
    unittest.main()
