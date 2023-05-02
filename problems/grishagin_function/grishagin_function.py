import numpy as np
import problems.grishagin_function.grishagin_generation as grishaginGen
import math


class GrishaginFunction:

    def __init__(self, function_number: int):
        self.dimension = 2
        self.icnf = np.ndarray(shape=(45,), dtype=np.dtype(int))
        self.af = np.ndarray(shape=(7, 7), dtype=np.double)
        self.bf = np.ndarray(shape=(7, 7), dtype=np.double)
        self.cf = np.ndarray(shape=(7, 7), dtype=np.double)
        self.df = np.ndarray(shape=(7, 7), dtype=np.double)
        if function_number < 1 or function_number > 100:
            function_number = 1
        self.fn = function_number
        self.SetFunctionNumber()

    def Calculate(self, x: np.ndarray(shape=(1), dtype=np.double)) -> np.double:
        """Compute selected function at given point."""
        snx = np.ndarray(shape=(7,), dtype=np.double)
        csx = np.ndarray(shape=(7,), dtype=np.double)
        sny = np.ndarray(shape=(7,), dtype=np.double)
        csy = np.ndarray(shape=(7,), dtype=np.double)

        d1 = math.pi * x[0]
        d2 = math.pi * x[1]
        sx1 = math.sin(d1)
        cx1 = math.cos(d1)
        sy1 = math.sin(d2)
        cy1 = math.cos(d2)
        snx[0] = sx1
        csx[0] = cx1
        sny[0] = sy1
        csy[0] = cy1

        for i in range(0, 6):
            snx[i + 1] = snx[i] * cx1 + csx[i] * sx1
            csx[i + 1] = csx[i] * cx1 - snx[i] * sx1
            sny[i + 1] = sny[i] * cy1 + csy[i] * sy1
            csy[i + 1] = csy[i] * cy1 - sny[i] * sy1

        d1 = 0
        d2 = 0

        for i in range(0, 7):
            for j in range(0, 7):
                d1 = d1 + self.af[i][j] * snx[i] * sny[j] + self.bf[i][j] * csx[i] * csy[j]
                d2 = d2 + self.cf[i][j] * snx[i] * sny[j] - self.df[i][j] * csx[i] * csy[j]

        value = -1 * math.sqrt(d1 * d1 + d2 * d2)

        return value

    def SetFunctionNumber(self):
        lst = 10
        i1 = int((self.fn - 1) / lst)
        i2 = int(i1 * lst)
        for j in range(len(grishaginGen.matcon[i1])):
            self.icnf[j] = int(grishaginGen.matcon[i1][j])

        if i2 != (self.fn - 1):
            i3 = self.fn - 1 - i2
            for j in range(1, int(i3 + 1)):
                for i in range(0, 196):
                    self.rndm20(self.icnf)

        for j in range(0, 7):
            for i in range(0, 7):
                self.af[i][j] = 2. * self.rndm20(self.icnf) - 1.
                self.cf[i][j] = 2. * self.rndm20(self.icnf) - 1.
        for j in range(0, 7):
            for i in range(0, 7):
                self.bf[i][j] = 2. * self.rndm20(self.icnf) - 1.
                self.df[i][j] = 2. * self.rndm20(self.icnf) - 1.

    def rndm20(self, k: np.array) -> np.double:
        k1 = np.ndarray(shape=(45,), dtype=np.dtype(int))
        for i in range(0, 38):
            k1[i] = k[i + 7]
        for i in range(38, 45):
            k1[i] = 0
        for i in range(0, 45):
            k[i] = abs(k[i] - k1[i])
        for i in range(27, 45):
            k1[i] = k[i - 27]
        for i in range(0, 27):
            k1[i] = 0
        self.gen(k, k1, 9, 44)
        self.gen(k, k1, 0, 8)

        rndm = 0.
        de2 = 1.
        for i in range(0, 36):
            de2 = de2 / 2
            rndm = rndm + k[i + 9] * de2

        return rndm

    def gen(self, k: np.array, k1: np.array, kap1: int, kap2: int):
        jct = 0
        for i in range(kap2, kap1 - 1, -1):
            j = int((k[i] + k1[i] + jct) / 2)
            k[i] = k[i] + k1[i] + jct - j * 2
            jct = j
        if jct != 0:
            for i in range(kap2, kap1 - 1, -1):
                j = int((k[i] + jct) / 2)
                k[i] = k[i] + jct - j * 2
                jct = j

    def GetOptimumPoint(self) -> np.ndarray:
        y = np.ndarray(shape=(self.dimension,), dtype=np.double)
        y[0] = grishaginGen.rand_minimums[2 * (self.fn - 1)]
        y[1] = grishaginGen.rand_minimums[2 * (self.fn - 1) + 1]
        return y
