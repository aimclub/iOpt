import unittest
import numpy as np

from iOpt.evolvent.evolvent import Evolvent


class TestEvolvent(unittest.TestCase):
    def setUp(self):
        self.ev1 = Evolvent([-1], [1])  # N = 1
        self.ev2 = Evolvent([-1, -1], [1, 1], 2, 10)  # N = 2, m = 10

    def test_Preimages_N1(self):
        y = [0]
        self.assertEqual(self.ev1.get_preimages(y), 0.5)

    def test_XtoYandBack_N1(self):
        x1 = 0.5
        y = self.ev1.get_image(x1)
        x2 = self.ev1.get_inverse_image(y)
        self.assertEqual(x1, x2)

    def test_Preimages_N2(self):
        y = [0.5, 0.5]
        self.assertEqual(self.ev2.get_preimages(y), 0.625)

    def test_XtoYandBack_N2(self):
        x1 = 0.625
        y = self.ev2.get_image(x1)
        x2 = self.ev2.get_inverse_image(y)
        self.assertEqual(x1, x2)

    def test_YtoXandBack_N2(self):
        y1 = np.array([0.5, 0.5])
        x = self.ev2.get_inverse_image(y1)
        y2 = self.ev2.get_image(x)
        np.testing.assert_array_almost_equal(y1, y2, decimal=3)
        # self.assertAlmostEqual(y1.tolist(), y2.tolist())

    def test_fileGetInverseImage(self):

        with open('test/evolventTestData/evolventGetInverseImage.txt') as file:
            for line in file:
                # читаем строку N = ; m =
                (Nstr, mstr) = line.split(';')
                # извлекаем N и m
                N = int(Nstr.split('=')[1])
                m = int(mstr.split('=')[1])
                # извлекаем значения y
                yValues = []
                for y in range(N):
                    yValues.append(file.readline())
                # извлекаем x
                xValue = file.readline()

                # создаем subtest для каждого набора x,y
                with self.subTest(yValues=yValues, xValue=xValue, N=N, m=m):
                    x = np.double(xValue.split('=')[1])
                    y = []
                    for yValue in yValues:
                        y.append(np.double(yValue.split('=')[1]))

                    # [-0.5, 0.5]
                    lower = - np.ones(N, dtype=np.int32) / 2
                    upper = np.ones(N, dtype=np.int32) / 2

                    evolvent = Evolvent(lower, upper, N, m)
                    xx = evolvent.get_inverse_image(y)

                    self.assertAlmostEqual(x, xx, 5)

    def test_fileGetImage(self):

        with open('test/evolventTestData/evolventGetImage.txt') as file:
            for line in file:
                # читаем строку N = ; m =
                (Nstr, mstr) = line.split(';')
                # извлекаем N и m
                N = int(Nstr.split('=')[1])
                m = int(mstr.split('=')[1])
                # извлекаем значения y
                yValues = []
                for y in range(N):
                    yValues.append(file.readline())
                # извлекаем x
                xValue = file.readline()

                # создаем subtest для каждого набора x,y
                with self.subTest(yValues=yValues, xValue=xValue, N=N, m=m):
                    x = np.double(xValue.split('=')[1])
                    y = []
                    for yValue in yValues:
                        y.append(np.double(yValue.split('=')[1]))

                    # [-0.5, 0.5]
                    lower = - np.ones(N, dtype=np.int32) / 2
                    upper = np.ones(N, dtype=np.int32) / 2

                    evolvent = Evolvent(lower, upper, N, m)
                    yy = evolvent.get_image(x)

                    np.testing.assert_array_almost_equal(y, yy, decimal=10)


# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()
