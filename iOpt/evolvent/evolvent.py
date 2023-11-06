import numpy as np
import math


class Evolvent:
    r"""Class Evolvent

    :param lower_bound_of_float_variables: array for lower bounds, А.
    :type  lower_bound_of_float_variables: np.ndarray(shape = (1), dtype = np.double).
    :param upper_bound_of_float_variables: array for upper bounds, В.
    :type  upper_bound_of_float_variables: np.ndarray(shape = (1), dtype = np.double).
    :param number_of_float_variables: dimension (N).
    :type  number_of_float_variables: int.
    :param evolvent_density: evolvent density (m).
    :type  evolvent_density: int.
    """

    def __init__(self,
                 lower_bound_of_float_variables: np.ndarray(shape=(1), dtype=np.double) = [],
                 upper_bound_of_float_variables: np.ndarray(shape=(1), dtype=np.double) = [],
                 number_of_float_variables: int = 1,
                 evolvent_density: int = 10
                 ):

        self.number_of_float_variables = number_of_float_variables
        self.lower_bound_of_float_variables = np.copy(lower_bound_of_float_variables)
        self.upper_bound_of_float_variables = np.copy(upper_bound_of_float_variables)
        self.evolvent_density = evolvent_density

        self.nexpValue = 0  # nexpExtended
        self.nexpExtended: np.double = 1.0

        # инициализируем массив y нулями
        self.yValues = np.zeros(self.number_of_float_variables, dtype=np.double)
        # np.ndarray(shape = (1), dtype = np.double) = [0,0] # y
        for i in range(0, self.number_of_float_variables):
            self.nexpExtended += self.nexpExtended

    # Установка границ
    # ----------------
    def set_bounds(self,
                   lower_bound_of_float_variables: np.ndarray(shape=(1), dtype=np.double) = [],
                   upper_bound_of_float_variables: np.ndarray(shape=(1), dtype=np.double) = []
                   ):
        r"""Set bounds

        :param lower_bound_of_float_variables: array for lower bounds, А.
        :type  lower_bound_of_float_variables: np.ndarray(shape = (1), dtype = np.double).
        :param upper_bound_of_float_variables: array for upper bounds, В.
        :type  upper_bound_of_float_variables: np.ndarray(shape = (1), dtype = np.double).
        """

        self.lower_bound_of_float_variables = np.copy(lower_bound_of_float_variables)
        self.upper_bound_of_float_variables = np.copy(upper_bound_of_float_variables)

    def get_image(self,
                  x: np.double
                  ) -> np.ndarray(shape=(1), dtype=np.double):
        r"""Get image (x->y)

        :param x: value of *x*.
        :type  x: np.double.
        :return: array of values *y*.
        :rtype: np.ndarray(shape = (1), dtype = np.double).

        """

        self.__get_y_on_x(x)
        self.__transform_p_2_d()
        return np.copy(self.yValues)

    def get_inverse_image(self,
                          y: np.ndarray(shape=(1), dtype=np.double)
                          ) -> np.double:
        r"""Get inverse image (y->x)

        :param y: value of *y*.
        :type  y: np.ndarray(shape = (1), dtype = np.double).
        :return: value of *x*.
        :rtype: np.double:.

        """
        self.yValues = np.copy(y)
        self.__transform_d_2_p()
        x = self.__get_x_on_y()
        return x

    # ----------------------
    def get_preimages(self,
                      y: np.ndarray(shape=(1), dtype=np.double),
                      ) -> np.double:
        r"""Get inverse image (y->x)

        :param y: value of *y*.
        :type  y: np.ndarray(shape = (1), dtype = np.double).
        :return: value of *x*.
        :rtype: np.double:.

        """
        self.yValues = np.copy(y)
        self.__transform_d_2_p()
        x = self.__get_x_on_y()
        return x

    # Преобразование
    # --------------------------------
    def __transform_p_2_d(self):
        for i in range(0, self.number_of_float_variables):
            self.yValues[i] = self.yValues[i] * (
                    self.upper_bound_of_float_variables[i] - self.lower_bound_of_float_variables[i]) + \
                              (self.upper_bound_of_float_variables[i] + self.lower_bound_of_float_variables[i]) / 2

    # Преобразование
    # --------------------------------
    def __transform_d_2_p(self):
        for i in range(0, self.number_of_float_variables):
            self.yValues[i] = (self.yValues[i] - (
                    self.upper_bound_of_float_variables[i] + self.lower_bound_of_float_variables[i]) / 2) / \
                              (self.upper_bound_of_float_variables[i] - self.lower_bound_of_float_variables[i])

    # ---------------------------------

    def __get_y_on_x(self, _x: np.double) -> np.ndarray(shape=(1), dtype=np.double):
        if self.number_of_float_variables == 1:
            self.yValues[0] = _x - 0.5
            return self.yValues

        iu: np.narray(shape=(1), dtype=np.int32)
        iv: np.narray(shape=(1), dtype=np.int32)
        node: np.int32
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

        # mn = self.evolvent_density * self.number_of_float_variables

        iw = np.ones(self.number_of_float_variables, dtype=np.int32)
        self.yValues = np.zeros(self.number_of_float_variables, dtype=np.double)
        iu = np.zeros(self.number_of_float_variables, dtype=np.int32)
        iv = np.zeros(self.number_of_float_variables, dtype=np.int32)

        for j in range(0, self.evolvent_density):
            if math.isclose(_x, 1.0):
                iis = self.nexpExtended - 1.0
                d = 0.0
            else:
                d *= self.nexpExtended
                iis = int(d)
                d -= iis

                # print(iis, self.number_of_float_variables)
            node = self.__calculate_node(iis, self.number_of_float_variables, iu, iv)
            # print(j, node)

            # заменить на () = () !
            i = iu[0]
            iu[0] = iu[it]
            iu[it] = i
            i = iv[0]
            iv[0] = iv[it]
            iv[it] = i

            if node == 0:
                node = it
            elif node == it:
                node = 0

            r *= 0.5
            it = node
            for i in range(0, self.number_of_float_variables):
                iu[i] *= iw[i]
                iw[i] *= -iv[i]
                self.yValues[i] += r * iu[i]

        return np.copy(self.yValues)

    # ---------------------------------
    def __get_x_on_y(self) -> np.double:
        x: np.double
        if self.number_of_float_variables == 1:
            x = self.yValues[0] + 0.5
            return x

        u: np.narray(shape=(1), dtype=np.int32)
        v: np.narray(shape=(1), dtype=np.int32)
        w: np.narray(shape=(1), dtype=np.int32)
        r: np.double = 0.0
        i: np.int32
        j: np.int32
        it: np.int32
        node: np.int32
        r1: np.double
        iis: np.double
        w = np.ones(self.number_of_float_variables, dtype=np.int32)
        u = np.zeros(self.number_of_float_variables, dtype=np.int32)
        v = np.zeros(self.number_of_float_variables, dtype=np.int32)
        r = 0.5
        r1 = 1.0
        x = 0.0
        it = 0

        for j in range(0, self.evolvent_density):
            r *= 0.5
            for i in range(0, self.number_of_float_variables):
                if self.yValues[i] < 0:
                    u[i] = -1
                else:
                    u[i] = 1

                self.yValues[i] -= r * u[i]
                u[i] *= w[i]

            i = u[0]
            u[0] = u[it]
            u[it] = i

            iis, node, v = self.__calculate_numbr(u, v)
            # print(u)
            # print(v)
            # print(iis, node)

            i = v[0]
            v[0] = v[it]
            v[it] = i

            for i in range(0, self.number_of_float_variables):
                w[i] *= -v[i]

            if node == 0:
                node = it
            elif node == it:
                node = 0

            it = node
            r1 = r1 / self.nexpExtended
            x += r1 * iis

        return x

    # -----------------------------------------------------------------------------------------
    def __calculate_numbr(self,
                          u: np.ndarray(shape=(1), dtype=np.int32),
                          v: np.ndarray(shape=(1), dtype=np.int32)
                          ):
        i = 0
        k1 = -1
        k2 = 0
        l1 = 0
        node = 0
        iis: np.double
        iff: np.double

        iff = self.nexpExtended
        iis = 0.0

        for i in range(0, self.number_of_float_variables):
            iff /= 2
            k2 = -k1 * u[i]
            v[i] = u[i]
            k1 = k2
            if k2 < 0:
                node1 = i
            else:
                iis += iff
                node = i

        if math.isclose(iis, 0.0):
            node = self.number_of_float_variables - 1
        else:
            v[self.number_of_float_variables - 1] = -v[self.number_of_float_variables - 1]
            if math.isclose(iis, self.nexpExtended - 1.0):
                node = self.number_of_float_variables - 1
            else:
                if node1 == self.number_of_float_variables - 1:
                    v[node] = -v[node]
                else:
                    node = node1
        s = iis

        return s, node, v

    # -----------------------------------------------------------------------------------------
    def __calculate_node(self,
                         iis: np.double,
                         n: int,
                         u: np.ndarray(shape=(1), dtype=np.int32),
                         v: np.ndarray(shape=(1), dtype=np.int32),
                         ):

        iq = 1
        n1 = n - 1
        node = 0
        if math.isclose(iis, 0.0):
            node = n1
            for i in range(0, n):
                u[i] = -1
                v[i] = -1
        elif math.isclose(iis, self.nexpExtended - 1.0):
            node = n1
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
                        node = i
                        iq = -1
                    iis -= iff
                    k2 = 1
                else:
                    k2 = -1
                    if math.isclose(iis, (iff - 1.0)) and not math.isclose(iis, 0.0):
                        node = i
                        iq = 1
                j = -k1 * k2
                v[i] = j
                u[i] = j
                k1 = k2
            v[node] = v[node] * iq
            v[n1] = -v[n1]
        return node

# -----------------------------------------------------------------------------------------
