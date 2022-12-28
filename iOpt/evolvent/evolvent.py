import numpy as np
import math


class Evolvent:
    """Класс разверток

    :param lowerBoundOfFloatVariables: массив для левых (нижних) границ, А.
    :type  lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double).
    :param upperBoundOfFloatVariables: массив для правых (верхних) границ, В.
    :type  upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double).
    :param numberOfFloatVariables: размерность задачи (N).
    :type  numberOfFloatVariables: int
    :param evolventDensity: плотность развертки (m).
    :type  evolventDensity: int
    """

    def __init__(self,
                 lowerBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double) = [],
                 upperBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double) = [],
                 numberOfFloatVariables: int = 1,
                 evolventDensity: int = 10
                 ):

        self.numberOfFloatVariables = numberOfFloatVariables
        self.lowerBoundOfFloatVariables = np.copy(lowerBoundOfFloatVariables)
        self.upperBoundOfFloatVariables = np.copy(upperBoundOfFloatVariables)
        self.evolventDensity = evolventDensity

        self.nexpValue = 0  # nexpExtended
        self.nexpExtended: np.double = 1.0

        # инициализируем массив y нулями
        self.yValues = np.zeros(self.numberOfFloatVariables, dtype=np.double)
        # np.ndarray(shape = (1), dtype = np.double) = [0,0] # y
        for i in range(0, self.numberOfFloatVariables):
            self.nexpExtended += self.nexpExtended

    # Установка границ
    # ----------------
    def SetBounds(self,
                  lowerBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double) = [],
                  upperBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double) = []
                  ):
        """Установка граничных значений

        :param lowerBoundOfFloatVariables: массив для левых (нижних) границ, А.
        :type  lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double).
        :param upperBoundOfFloatVariables: массив для правых (верхних) границ, В.
        :type  upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double).
        """

        self.lowerBoundOfFloatVariables = np.copy(lowerBoundOfFloatVariables)
        self.upperBoundOfFloatVariables = np.copy(upperBoundOfFloatVariables)

    def GetImage(self,
                 x: np.double
                 ) -> np.ndarray(shape=(1), dtype=np.double):
        """Получить образ (x->y)

        :param x: значение x.
        :type  x: np.double.
        :return: массив значений *y*
        :rtype: np.ndarray(shape = (1), dtype = np.double).

        """

        self.__GetYonX(x)
        self.__TransformP2D()
        return np.copy(self.yValues)

    def GetInverseImage(self,
                        y: np.ndarray(shape=(1), dtype=np.double)
                        ) -> np.double:
        """Получить обратное значение образа (y->x)

        :param y: значение y.
        :type  y: np.ndarray(shape = (1), dtype = np.double)
        :return: значение *x*
        :rtype: np.double:.

        """
        self.yValues = np.copy(y)
        self.__TransformD2P()
        x = self.__GetXonY()
        return x

    # ----------------------
    def GetPreimages(self,
                     y: np.ndarray(shape=(1), dtype=np.double),
                     ) -> np.double:
        """Получить обратное значение образа (y->x)

        :param y: значение y.
        :type  y: np.ndarray(shape = (1), dtype = np.double)
        :return: значение *x*
        :rtype: np.double:.

        """
        self.yValues = np.copy(y)
        self.__TransformD2P()
        x = self.__GetXonY()
        return x

    # Преобразование
    # --------------------------------
    def __TransformP2D(self):
        for i in range(0, self.numberOfFloatVariables):
            self.yValues[i] = self.yValues[i] * (
                        self.upperBoundOfFloatVariables[i] - self.lowerBoundOfFloatVariables[i]) + \
                        (self.upperBoundOfFloatVariables[i] + self.lowerBoundOfFloatVariables[i]) / 2

    # Преобразование
    # --------------------------------
    def __TransformD2P(self):
        for i in range(0, self.numberOfFloatVariables):
            self.yValues[i] = (self.yValues[i] - (
                        self.upperBoundOfFloatVariables[i] + self.lowerBoundOfFloatVariables[i]) / 2) / \
                        (self.upperBoundOfFloatVariables[i] - self.lowerBoundOfFloatVariables[i])

    # ---------------------------------

    def __GetYonX(self, _x: np.double) -> np.ndarray(shape=(1), dtype=np.double):
        if self.numberOfFloatVariables == 1:
            self.yValues[0] = _x - 0.5
            return self.yValues

        iu: np.narray(shape=(1), dtype=np.int32)
        iv: np.narray(shape=(1), dtype=np.int32)
        l: np.int32
        d: np.double = 0.0
        # mn: np.int32
        r: np.double
        iw: np.narray(shape=(1), dtype=np.int32)
        it: np.int32
        i: np.int32
        j: np.int32
        iis: np.double

        d = _x
        r = 0.5
        it = 0

        # mn = self.evolventDensity * self.numberOfFloatVariables

        iw = np.ones(self.numberOfFloatVariables, dtype=np.int32)
        self.yValues = np.zeros(self.numberOfFloatVariables, dtype=np.double)
        iu = np.zeros(self.numberOfFloatVariables, dtype=np.int32)
        iv = np.zeros(self.numberOfFloatVariables, dtype=np.int32)

        for j in range(0, self.evolventDensity):
            if math.isclose(_x, 1.0):
                iis = self.nexpExtended - 1.0
                d = 0.0
            else:
                d *= self.nexpExtended
                iis = int(d)
                d -= iis

                # print(iis, self.numberOfFloatVariables)
            l = self.__CalculateNode(iis, self.numberOfFloatVariables, iu, iv)
            # print(j, l)

            # заменить на () = () !
            i = iu[0]
            iu[0] = iu[it]
            iu[it] = i
            i = iv[0]
            iv[0] = iv[it]
            iv[it] = i

            if l == 0:
                l = it
            elif l == it:
                l = 0

            r *= 0.5
            it = l
            for i in range(0, self.numberOfFloatVariables):
                iu[i] *= iw[i]
                iw[i] *= -iv[i]
                self.yValues[i] += r * iu[i]

        return np.copy(self.yValues)

    # ---------------------------------
    def __GetXonY(self) -> np.double:
        x: np.double
        if self.numberOfFloatVariables == 1:
            x = self.yValues[0] + 0.5
            return x

        u: np.narray(shape=(1), dtype=np.int32)
        v: np.narray(shape=(1), dtype=np.int32)
        w: np.narray(shape=(1), dtype=np.int32)
        r: np.double = 0.0
        i: np.int32
        j: np.int32
        it: np.int32
        l: np.int32
        r1: np.double
        iis: np.double
        w = np.ones(self.numberOfFloatVariables, dtype=np.int32)
        u = np.zeros(self.numberOfFloatVariables, dtype=np.int32)
        v = np.zeros(self.numberOfFloatVariables, dtype=np.int32)
        r = 0.5
        r1 = 1.0
        x = 0.0
        it = 0

        for j in range(0, self.evolventDensity):
            r *= 0.5
            for i in range(0, self.numberOfFloatVariables):
                if self.yValues[i] < 0:
                    u[i] = -1
                else:
                    u[i] = 1

                self.yValues[i] -= r * u[i]
                u[i] *= w[i]

            i = u[0]
            u[0] = u[it]
            u[it] = i

            iis, l, v = self.__CalculateNumbr(u, v)
            # print(u)
            # print(v)
            # print(iis, l)

            i = v[0]
            v[0] = v[it]
            v[it] = i

            for i in range(0, self.numberOfFloatVariables):
                w[i] *= -v[i]

            if l == 0:
                l = it
            elif l == it:
                l = 0

            it = l
            r1 = r1 / self.nexpExtended
            x += r1 * iis

        return x

    # -----------------------------------------------------------------------------------------
    def __CalculateNumbr(self,
                         u: np.ndarray(shape=(1), dtype=np.int32),
                         v: np.ndarray(shape=(1), dtype=np.int32)
                         ):
        i = 0
        k1 = -1
        k2 = 0
        l1 = 0
        l = 0
        iis: np.double
        iff: np.double

        iff = self.nexpExtended
        iis = 0.0

        for i in range(0, self.numberOfFloatVariables):
            iff /= 2
            k2 = -k1 * u[i]
            v[i] = u[i]
            k1 = k2
            if k2 < 0:
                l1 = i
            else:
                iis += iff
                l = i

        if math.isclose(iis, 0.0):
            l = self.numberOfFloatVariables - 1
        else:
            v[self.numberOfFloatVariables - 1] = -v[self.numberOfFloatVariables - 1]
            if math.isclose(iis, self.nexpExtended - 1.0):
                l = self.numberOfFloatVariables - 1
            else:
                if l1 == self.numberOfFloatVariables - 1:
                    v[l] = -v[l]
                else:
                    l = l1
        s = iis

        return s, l, v

    # -----------------------------------------------------------------------------------------
    def __CalculateNode(self,
                        iis: np.double,
                        n: int,
                        u: np.ndarray(shape=(1), dtype=np.int32),
                        v: np.ndarray(shape=(1), dtype=np.int32),
                        ):

        iq = 1
        n1 = n - 1
        l = 0
        if math.isclose(iis, 0.0):
            l = n1
            for i in range(0, n):
                u[i] = -1
                v[i] = -1
        elif math.isclose(iis, self.nexpExtended - 1.0):
            l = n1
            u[0] = 1
            v[0] = 1
            for i in range(1, n):
                u[i] = -1
                v[i] = -1
            v[n1] = 1
        else:
            iff = self.nexpExtended
            k1 = -1
            for i in range(0, n):
                iff /= 2
                if iis >= iff:  # исправить сравнение!
                    if math.isclose(iis, iff) and not math.isclose(iis, 1.0):
                        l = i
                        iq = -1
                    iis -= iff
                    k2 = 1
                else:
                    k2 = -1
                    if math.isclose(iis, (iff - 1.0)) and not math.isclose(iis, 0.0):
                        l = i
                        iq = 1
                j = -k1 * k2
                v[i] = j
                u[i] = j
                k1 = k2
            v[l] = v[l] * iq
            v[n1] = -v[n1]
        return l

# -----------------------------------------------------------------------------------------
