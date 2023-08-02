import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Point


from problems.ex1222 import ex1222
from problems.Floudas import Floudas
from problems.g8c import g8c
from problems.gbd import gbd
from problems.GKLS import GKLS
from problems.grishagin import Grishagin
from problems.nvs21 import nvs21
from problems.p1 import p1
from problems.p2 import p2
from problems.p7 import p7
from problems.Pern import Pern
from problems.rastrigin import Rastrigin
from problems.rastriginInt import RastriginInt
from problems.romeijn1c import Romeijn1c
from problems.romeijn2c import Romeijn2c
from problems.romeijn3c import Romeijn3c
from problems.romeijn5c import Romeijn5c
from problems.stronginc2 import Stronginc2
from problems.stronginc3 import Stronginc3
from problems.stronginc5 import Stronginc5
from problems.shekel4 import Shekel4
from problems.Synthes import Synthes
from problems.Yuan import Yuan



import test.problems.pointsForTest as pft
from test.problems.pointsForTest import ex1222_points, Floudas_points, g8c_points, gbd_points, gkls_points, grishagin_points,\
    nvs21_points, p1_points, p2_points, p7_points, \
    Pern_points, rastrigin_points, rastriginInt_points, romeijn1c_points, romeijn2c_points, romeijn3c_points, \
    romeijn5с_points, stronginc2_points, stronginc3_points, stronginc5_points, \
    shekel4_points, Synthes_points, Yuan_points


class TestAllProblems(unittest.TestCase):

    def test_ex1222(self):
        problem = ex1222()
        sample = pft.ex1222_points
        self.Calculate(problem, sample)

    def test_Floudas(self):
        problem = Floudas()
        sample = pft.Floudas_points
        self.Calculate(problem, sample)

    def test_g8c(self):
        problem = g8c()
        sample = pft.g8c_points
        self.Calculate(problem, sample)

    def test_gbd(self):
        problem = gbd()
        sample = pft.gbd_points
        self.Calculate(problem, sample)

    def test_GKLS(self):
        problem = GKLS(3,1)
        sample = pft.gkls_points
        self.Calculate(problem, sample)

    def test_Grishagin(self):
        problem = Grishagin(1)
        sample = pft.grishagin_points
        self.Calculate(problem, sample)

    def test_nvs21(self):
        problem = nvs21()
        sample = pft.nvs21_points
        self.Calculate(problem, sample)

    def test_p1(self):
        problem = p1()
        sample = pft.p1_points
        self.Calculate(problem, sample)

    def test_p2(self):
        problem = p2()
        sample = pft.p2_points
        self.Calculate(problem, sample)

    def test_p7(self):
        problem = p7()
        sample = pft.p7_points
        self.Calculate(problem, sample)

    def test_Pern(self):
        problem = Pern()
        sample = pft.Pern_points
        self.Calculate(problem, sample)

    def test_Rastrigin(self):
        problem = Rastrigin(2)
        sample = pft.rastrigin_points
        self.Calculate(problem, sample)

    def test_RastriginInt(self):
        problem = RastriginInt(3, 2)
        sample = pft.rastriginInt_points
        self.Calculate(problem, sample)

    def test_Romeijn1c(self):
        problem = Romeijn1c()
        sample = pft.romeijn1c_points
        self.Calculate(problem, sample)

    def test_Romeijn2c(self):
        problem = Romeijn2c()
        sample = pft.romeijn2c_points
        self.Calculate(problem, sample)

    def test_Romeijn3c(self):
        problem = Romeijn3c()
        sample = pft.romeijn3c_points
        self.Calculate(problem, sample)

    def test_Romeijn5c(self):
        problem = Romeijn5c()
        sample = pft.romeijn5с_points
        self.Calculate(problem, sample)

    def test_Stronginc2(self):
        problem = Stronginc2()
        sample = pft.stronginc2_points
        self.Calculate(problem, sample)

    def test_Stronginc3(self):
        problem = Stronginc3()
        sample = pft.stronginc3_points
        self.Calculate(problem, sample)

    def test_Stronginc5(self):
        problem = Stronginc5()
        sample = pft.stronginc5_points
        self.Calculate(problem, sample)

    def test_Shekel4(self):
        problem = Shekel4(1)
        sample = pft.shekel4_points
        self.Calculate(problem, sample)

    def test_Synthes(self):
        problem = Synthes()
        sample = pft.Synthes_points
        self.Calculate(problem, sample)

    def test_Yuan(self):
        problem = Yuan()
        sample = pft.Yuan_points
        self.Calculate(problem, sample)

    def Calculate(self, problem, Sample):
        for i in range(0, len(Sample.test_points)):
            fv_point = []
            for j in range(0, problem.number_of_float_variables):
                fv_point.append(np.double(Sample.test_points[i][j]))
            dv_point = []
            for j in range(problem.number_of_float_variables, problem.dimension):
                dv_point.append(Sample.test_points[i][j])

            point = Point(fv_point, dv_point)
            function_value = FunctionValue()

            if problem.number_of_constraints > 0:
                if Sample.test_points[i][problem.dimension] == problem.number_of_constraints:
                    function_value.type = FunctionType.OBJECTIV
                else:
                    function_value.type = FunctionType.CONSTRAINT
                    function_value.functionID = Sample.test_points[i][problem.dimension]

            function_value = problem.calculate(point, function_value)

            if problem.number_of_constraints > 0:
                self.assertAlmostEqual(function_value.value,
                                       np.double(Sample.test_points[i][problem.dimension+1]), 5)
            else:
                self.assertAlmostEqual(function_value.value,
                                       np.double(Sample.test_points[i][problem.dimension]), 5)

        print(problem.name, "is OK")


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
