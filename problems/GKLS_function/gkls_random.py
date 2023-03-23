import numpy as np
import math


class GKLSRandomGenerator:
    KK = 100  # the long lag
    LL = 37  # the short lag */

    TT = 70  # guaranteed separation between streams */

    QUALITY = 1009  # recommended quality level for high-res use */

    NUM_RND = 1009  # size of the array of random numbers */

    @staticmethod
    def mod_sum(x, y):
        return (((x) + (y)) - (int)((x) + (y)))  # (x+y) mod 1.0 */

    @staticmethod
    def is_odd(s: int):
        return ((s) & 1)

    def __init__(self):
        self.rnd_num = np.zeros(GKLSRandomGenerator.KK, dtype=np.double)  # array of random numbers */
        self.ran_u = np.zeros(GKLSRandomGenerator.LL, dtype=np.double)  # the generator state */

    def Initialize(self, seed,
                   rnd_num_mem: np.ndarray(shape=(1), dtype=np.double),
                   rand_condition_mem: np.ndarray(shape=(1), dtype=np.double)):
        self.ran_u = rand_condition_mem
        self.rnd_num = rnd_num_mem
        j = 0
        t = 0
        s = 0

        u = np.zeros([GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 1], dtype=np.double)
        ul = np.zeros([GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 1], dtype=np.double)

        ulp = (1.0 / (1 << 30)) / (1 << 22)  # 2 to the -52 * /
        ss = 2.0 * ulp * ((seed & 0x3fffffff) + 2)

        for j in range(GKLSRandomGenerator.KK):
            u[j] = ss
            ul[j] = 0.0  # bootstrap the buffer */
            ss += ss
            if ss >= 1.0:
                ss -= 1.0 - 2 * ulp  # cyclic shift of 51 bits */

        for j in range(GKLSRandomGenerator.KK, GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 1):
            u[j] = ul[j] = 0.0

        u[1] += ulp
        ul[1] = ulp  # make u[1] (and only u[1]) "odd" */
        s = seed & 0x3fffffff
        t = GKLSRandomGenerator.TT - 1

        while (t > 0):
            for j in range(GKLSRandomGenerator.KK - 1, 0, -1):
                ul[j + j] = ul[j]
                u[j + j] = u[j]  # "square" */

            for j in range(GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 2,
                           GKLSRandomGenerator.KK - GKLSRandomGenerator.LL, -2):
                ul[GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 1 - j] = 0.0
                u[GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 1 - j] = u[j] - ul[j]

            for j in range(GKLSRandomGenerator.KK + GKLSRandomGenerator.KK - 2, GKLSRandomGenerator.KK - 1, -1):
                if (ul[j] != 0):
                    ul[j - (GKLSRandomGenerator.KK - GKLSRandomGenerator.LL)] = ulp - ul[
                        j - (GKLSRandomGenerator.KK - GKLSRandomGenerator.LL)]
                    u[j - (GKLSRandomGenerator.KK - GKLSRandomGenerator.LL)] = GKLSRandomGenerator.mod_sum(
                        u[j - (GKLSRandomGenerator.KK - GKLSRandomGenerator.LL)], u[j])
                    ul[j - GKLSRandomGenerator.KK] = ulp - ul[j - GKLSRandomGenerator.KK]
                    u[j - GKLSRandomGenerator.KK] = GKLSRandomGenerator.mod_sum(u[j - GKLSRandomGenerator.KK], u[j])

            if (GKLSRandomGenerator.is_odd(s) != 0):  # "multiply by z" */
                for j in range(GKLSRandomGenerator.KK, 0, -1):
                    ul[j] = ul[j - 1]
                    u[j] = u[j - 1]
                ul[0] = ul[GKLSRandomGenerator.KK]
                u[0] = u[GKLSRandomGenerator.KK]  # shift the buffer cyclically */
                if (ul[GKLSRandomGenerator.KK]):
                    ul[GKLSRandomGenerator.LL] = ulp - ul[GKLSRandomGenerator.LL]
                    u[GKLSRandomGenerator.LL] = GKLSRandomGenerator.mod_sum(u[GKLSRandomGenerator.LL],
                                                                            u[GKLSRandomGenerator.KK])

            if (s):
                s >>= 1
            else:
                t = t - 1

        for j in range(GKLSRandomGenerator.LL):
            self.ran_u[j + GKLSRandomGenerator.KK - GKLSRandomGenerator.LL] = u[j]
        for j in range(GKLSRandomGenerator.KK):
            self.ran_u[j - GKLSRandomGenerator.LL] = u[j]

    def GenerateNextNumbers(self):
        i = 0
        j = 0
        n = GKLSRandomGenerator.NUM_RND
        for j in range(GKLSRandomGenerator.KK):
            self.rnd_num[j] = self.ran_u[j]
        for j in range(GKLSRandomGenerator.KK, n):
            self.rnd_num[j] = GKLSRandomGenerator.mod_sum(self.rnd_num[j - GKLSRandomGenerator.KK],
                                                          self.rnd_num[j - GKLSRandomGenerator.LL])
        j = j + 1

        for i in range(GKLSRandomGenerator.LL):
            self.ran_u[i] = GKLSRandomGenerator.mod_sum(self.rnd_num[j - GKLSRandomGenerator.KK],
                                                        self.rnd_num[j - GKLSRandomGenerator.LL])
            j = j + 1

        for i in range(GKLSRandomGenerator.LL, GKLSRandomGenerator.KK):
            self.ran_u[i] = GKLSRandomGenerator.mod_sum(self.rnd_num[j - GKLSRandomGenerator.KK],
                                                        self.ran_u[i - GKLSRandomGenerator.LL])
            j = j + 1

    def GetRandomNumber(self, indx):
        return self.rnd_num[indx]

    def GetGeneratorState(self, indx):
        return self.ran_u[indx]
