from math import isclose


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

    def __init__(self, lowerBoundOfFloatVariables: list[float] = [], upperBoundOfFloatVariables: list[float] = [],
                 numberOfFloatVariables: int = 1, evolventDensity: int = 10):

        self.numberOfFloatVariables = numberOfFloatVariables
        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables.copy()
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables.copy()
        self.evolventDensity = evolventDensity

        self.nexpValue = 0  # nexpExtended
        self.nexpExtended: float = 2 ** self.numberOfFloatVariables

        self.yValues = [0.0] * len(self.lowerBoundOfFloatVariables)

    def SetBounds(self, lowerBoundOfFloatVariables: list[float] = [],
                  upperBoundOfFloatVariables: list[float] = []) -> None:
        """Установка граничных значений

        :param lowerBoundOfFloatVariables: массив для левых (нижних) границ, А.
        :type  lowerBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double).
        :param upperBoundOfFloatVariables: массив для правых (верхних) границ, В.
        :type  upperBoundOfFloatVariables: np.ndarray(shape = (1), dtype = np.double).
        """

        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables.copy()
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables.copy()

    def GetImage(self, x: float) -> list[float]:
        """Получить образ (x->y)

        :param x: значение x.
        :type  x: np.double.
        :return: массив значений *y*
        :rtype: np.ndarray(shape = (1), dtype = np.double).

        """

        self.__GetYonX(x)
        self.__TransformP2D()
        return self.yValues.copy()

    def GetInverseImage(self, y: list[float]) -> float:
        """Получить обратное значение образа (y->x)

        :param y: значение y.
        :type  y: np.ndarray(shape = (1), dtype = np.double)
        :return: значение *x*
        :rtype: np.double:.

        """
        self.yValues = y.copy()
        self.__TransformD2P()
        x = self.__GetXonY()
        return x

    def GetPreimages(self, y: list[float]) -> float:
        """Получить обратное значение образа (y->x)

        :param y: значение y.
        :type  y: np.ndarray(shape = (1), dtype = np.double)
        :return: значение *x*
        :rtype: np.double:.

        """
        return self.GetInverseImage(y)

    def __TransformP2D(self) -> None:
        self.yValues = [y * (upper - lower) + (upper + lower) / 2
                        for y, upper, lower in zip(self.yValues,
                                                   self.upperBoundOfFloatVariables,
                                                   self.lowerBoundOfFloatVariables)]

    def __TransformD2P(self) -> None:
        self.yValues = [(y - 0.5 * (upper + lower)) / (upper - lower)
                        for y, upper, lower in zip(self.yValues,
                                                   self.upperBoundOfFloatVariables,
                                                   self.lowerBoundOfFloatVariables)]

    def __GetYonX(self, x: float) -> None:
        if self.numberOfFloatVariables == 1:

            self.yValues[0] = x - 0.5
            return

        d, it = x, 0
        r = 0.5

        # mn = self.evolventDensity * self.numberOfFloatVariables

        iw = [1] * self.numberOfFloatVariables
        iu = [0] * self.numberOfFloatVariables
        iv = [0] * self.numberOfFloatVariables
        self.yValues = [0] * self.numberOfFloatVariables

        for _ in range(self.evolventDensity):
            if isclose(x, 1):
                iis = self.nexpExtended - 1
                d = 0.0
            else:
                d *= self.nexpExtended
                iis = int(d)
                d -= iis

                # print(iis, self.numberOfFloatVariables)
            node = self.__CalculateNode(iis, self.numberOfFloatVariables, iu, iv)
            node = node + it * (node == 0) - it * (node == it)

            iu[0], iu[it] = iu[it], iu[0]
            iv[0], iv[it] = iv[it], iv[0]

            r /= 2
            it = node

            for i in range(self.numberOfFloatVariables):
                iu[i] *= iw[i]
                iw[i] *= -iv[i]
                self.yValues[i] += r * iu[i]

    def __GetXonY(self) -> float:

        if self.numberOfFloatVariables == 1:
            return self.yValues[0] + 0.5

        w = [1] * self.numberOfFloatVariables
        u = [0] * self.numberOfFloatVariables
        v = [0] * self.numberOfFloatVariables

        r, r1, x, = 0.5, 1.0, 0.0
        it = 0

        for _ in range(self.evolventDensity):

            r *= 0.5

            for i in range(self.numberOfFloatVariables):

                u[i] = 2 * (self.yValues[i] >= 0) - 1
                self.yValues[i] -= r * u[i]
                u[i] *= w[i]

            u[0], u[it] = u[it], u[0]

            iis, node, v = self.__CalculateNumbr(u, v)
            node = node + it * (node == 0) - it * (node == it)

            v[0], v[it] = v[it], v[0]

            w = [-1 * x * y for x, y in zip(w, v)]

            it = node
            r1 /= self.nexpExtended
            x += r1 * iis

        return x

    def __CalculateNumbr(self, u: list[int], v: list[int]) -> tuple[float, int, list[int]]:
        iff, iis = self.nexpExtended, 0.0
        k1, node1, node = -1.0, 0, 0

        for i in range(self.numberOfFloatVariables):
            iff /= 2
            v[i] = u[i]
            k1 = -k1 * u[i]
            if k1 < 0:
                node1 = i
            else:
                iis += iff
                node = i

        if isclose(iis, 0.0):
            return iis, self.numberOfFloatVariables - 1, v
        v[-1] *= -1
        if isclose(iis, self.nexpExtended - 1):
            node = self.numberOfFloatVariables - 1
        elif node1 == self.numberOfFloatVariables - 1:
            v[node] *= -1
        else:
            node = node1
        return iis, node, v

    def __CalculateNode(self, iis: float, n: int, u: list[int], v: list[int]) -> int:

        node, iq = 0, 1
        if isclose(iis, 0.0):
            node = n - 1
            u[:] = [-1] * n
            v[:] = [-1] * n
        elif isclose(iis, self.nexpExtended - 1):
            node = n - 1
            u[:] = [1] + [-1] * (n - 1)
            v[:] = [1] + [-1] * (n - 2) + [1]
        else:
            iff, k1 = self.nexpExtended, -1
            for i in range(n):
                iff /= 2
                k2 = 2 * (iis >= iff) - 1  # исправить сравнение!
                if k2 == 1:
                    if isclose(iis, iff) and not isclose(iis, 1):
                        node, iq = i, -k2
                    iis -= iff
                elif isclose(iis, iff - 1) and not isclose(iis, 0):
                    node, iq = i, -k2
                j = -k1 * k2
                v[i] = u[i] = j
                k1 = k2
            v[node] *= iq
            v[-1] *= -1
        return node
