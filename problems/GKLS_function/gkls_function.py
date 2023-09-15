import numpy as np
from problems.GKLS_function.gkls_random import GKLSRandomGenerator


# ***************************************************************************#
#        GKLS-Generator of Classes of ND  (non-differentiable),              #
#                                 D  (continuously differentiable), and      #
#                                 D2 (twice continuously differentiable)     #
#                     Test Functions for Global Optimization                 #
#                                                                            #
#   Authors:                                                                 #
#                                                                            #
#   M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev                         #
#                                                                            #
#   (C) 2002-2005                                                            #
#                                                                            #
#   References:                                                              #
#                                                                            #
#   1. M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev (2003),              #
#   Algorithm 829: Software for Generation of Classes of Test Functions      #
#   with Known Local and Global Minima for Global Optimization.              #
#   ACM Transactions on Mathematical Software, Vol. 29, no. 4, pp. 469-480.  #
#                                                                            #
#   2. D.Knuth (1997), The Art of Computer Programming, Vol. 2:              #
#   Seminumerical Algorithms (Third Edition). Reading, Massachusetts:        #
#   Addison-Wesley.                                                          #
#                                                                            #
#   The software constructs a convex quadratic function (paraboloid) and then#
#   systematically distorts randomly selected parts of this function         #
#   by polynomials in order to introduce local minima and to construct test  #
#   functions which are non-differentiable (ND-type), continuously           #
#   differentiable (D-type), and twice continuously differentiable (D2-type) #
#   at the feasible region.                                                  #
#                                                                            #
#   Each test class is defined by the following parameters:                  #
#  (1) problem dimension                                                     #
#  (2) number of local minima including the paraboloid min and the global min#
#  (3) global minimum value                                                  #
#  (3) distance from the paraboloid vertex to the global minimizer           #
#  (4) radius of the attraction region of the global minimizer               #
# ***************************************************************************#

class T_GKLS_Minima:
    def __init__(self):
        self.f = np.zeros(1, dtype=np.double)  # list of local minima values *#
        self.w_rho = np.zeros(1, dtype=np.double)  # list of radius weights *#
        self.peak = np.zeros(1, dtype=np.double)  # list of parameters gamma(i) = *#
        #  = local minimum value(i) - paraboloid *#
        #    minimum within attraction regions *#
        #    of local minimizer(i)
        #    *#
        rho = np.zeros(1, dtype=np.double)  # list of attraction regions radii *#
        local_min = np.zeros((1, 1), dtype=np.double)  # list of local minimizers coordinates *#


class T_GKLS_GlobalMinima:
    def __init__(self):
        self.num_global_minima = 0  # number of global minima *#
        self.gm_index = np.zeros(1, dtype=int)  # list of indices of generated *#


class GKLSClass:
    Hard = 0
    Simple = 1


class GKLSFuncionType:
    TND = 0
    TD = 1
    TD2 = 2


class GKLSParameters:
    def __init__(self,
                 _dimension,
                 _globalMinimumValue,
                 _numberOfLocalMinima,
                 _globalDistance,
                 _globalRadius,
                 _GKLStype):
        self.dimension = _dimension
        self.globalMinimumValue = _globalMinimumValue
        self.numberOfLocalMinima = _numberOfLocalMinima
        self.globalDistance = _globalDistance
        self.globalRadius = _globalRadius
        self.GKLStype = _GKLStype


class GKLSFunction:
    # Penalty value of the generated function if x is not in D *#
    GKLS_MAX_VALUE = 1E+100

    # Value of the machine zero in the floating-point arithmetic *#
    GKLS_PRECISION = 1.0E-10

    # Default value of the paraboloid minimum *#
    GKLS_PARABOLOID_MIN = 0.0

    # Global minimum value: to be less than GKLS_PARABOLOID_MIN *#
    GKLS_GLOBAL_MIN_VALUE = -1.0

    # Max value of the parameter delta for the D2-type class function *#
    # The parameter delta is chosen randomly from (0, GKLS_DELTA_MAX_VALUE )
    # *#
    GKLS_DELTA_MAX_VALUE = 10.0

    # Constant pi *#

    PI = 3.14159265

    # Error codes *#
    GKLS_OK = 0
    GKLS_DIM_ERROR = 1
    GKLS_NUM_MINIMA_ERROR = 2
    GKLS_FUNC_NUMBER_ERROR = 3
    GKLS_BOUNDARY_ERROR = 4
    GKLS_GLOBAL_MIN_VALUE_ERROR = 5
    GKLS_GLOBAL_DIST_ERROR = 6
    GKLS_GLOBAL_RADIUS_ERROR = 7
    GKLS_MEMORY_ERROR = 8
    GKLS_DERIV_EVAL_ERROR = 9

    # Reserved error codes *#
    GKLS_GREAT_DIM = 10
    GKLS_RHO_ERROR = 11
    GKLS_PEAK_ERROR = 12
    GKLS_GLOBAL_BASIN_INTERSECTION = 13

    # Internal error codes *#
    GKLS_PARABOLA_MIN_COINCIDENCE_ERROR = 14
    GKLS_LOCAL_MIN_COINCIDENCE_ERROR = 15
    GKLS_FLOATING_POINT_ERROR = 16

    def __init__(self):
        self.mFunctionNumber = 0
        self.mDimension = 0
        self.mIsGeneratorMemoryAllocated = False
        self.mIsDomainMemeoryAllocated = False
        self.mFunctionType = GKLSFuncionType.TD

        self.mRndGenerator = GKLSRandomGenerator()

        # -------------- Variables accessible by the user ---------------------
        # *#
        self.GKLS_domain_left = np.zeros(1, dtype=np.double)  # left boundary vector of D *#
        # D=[GKLS_domain_left; GKLS_domain_ight] *#
        self.GKLS_domain_right = np.zeros(1, dtype=np.double)  # right boundary vector of D *#

        self.GKLS_dim = 0  # dimension of the problem, *#
        # 2<=test_dim<NUM_RND (see random) *#
        self.GKLS_num_minima = 0  # number of local minima, >=2 *#

        self.GKLS_global_dist = np.double(0)  # distance from the paraboloid minimizer *#
        # to the global minimizer *#
        self.GKLS_global_radius = np.double(0)  # radius of the global minimizer *#
        # attraction region *#
        self.GKLS_global_value = np.double(0)  # global minimum value, *#
        # test_global_value < GKLS_PARABOLOID_MIN *#
        self.GKLS_minima = T_GKLS_Minima()
        # see the structures type description *#
        self.GKLS_glob = T_GKLS_GlobalMinima()

        # --------------------------- Global variables ----------------------*#
        self.isArgSet = 0  # isArgSet == 1 if all necessary parameters are set *#

        self.delta = np.double(0)
        # parameter using in D2-type function generation; *#
        # it is chosen randomly from the *#
        # open interval (0,GKLS_DELTA_MAX_VALUE) *#
        self.rnd_counter = 0  # index of random array elements *#

        self.rnd_num = np.zeros(GKLSRandomGenerator.NUM_RND, dtype=np.double)
        self.rand_condition = np.zeros(GKLSRandomGenerator.KK, dtype=np.double)

    def Calculate(self, x: np.ndarray(shape=(1), dtype=np.double)):
        value = self.CalculateDFunction(x)
        # switch(mFunctionType)
        # {
        # case TND:
        #  value = CalculateNDFunction(x);
        #
        # case TD:
        #  value = CalculateDFunction(x);
        #
        # case TD2:
        #  value = CalculateD2Function(x);
        return value

    def GKLS_domain_free(self):
        self.mIsDomainMemeoryAllocated = False

    def GKLS_free(self):
        self.isArgSet = 0
        self.mIsGeneratorMemoryAllocated = False

    def CalculateDFunction(self, x: np.ndarray(shape=(1), dtype=np.double)):
        i = 0
        index = 0
        norm = np.double(0)
        scal = np.double(0)
        a = np.double(0)
        rho = np.double(0)  # * working variables *#

        if not self.isArgSet:
            return GKLSFunction.GKLS_MAX_VALUE

        for i in range(self.GKLS_dim):
            if ((x[i] < self.GKLS_domain_left[i] - GKLSFunction.GKLS_PRECISION) or (
                    x[i] > self.GKLS_domain_right[i] + GKLSFunction.GKLS_PRECISION)):
                return GKLSFunction.GKLS_MAX_VALUE
        # Check wether x belongs to some basin of local minima, M(index) <> T *#
        # Attention: number of local minima must be >= 2 *#
        index = 1
        while ((index < self.GKLS_num_minima) and (
                self.GKLS_norm(self.GKLS_minima.local_min[index], x) > self.GKLS_minima.rho[index])):
            index = index + 1

        if (index == self.GKLS_num_minima):
            norm = self.GKLS_norm(self.GKLS_minima.local_min[0], x)
            # * Return the value of the paraboloid function *#
            return (norm * norm + self.GKLS_minima.f[0])

        # Check wether x coincides with the local minimizer M(index) *#
        if (self.GKLS_norm(x, self.GKLS_minima.local_min[index]) < GKLSFunction.GKLS_PRECISION):
            return self.GKLS_minima.f[index]

        norm = self.GKLS_norm(self.GKLS_minima.local_min[0], self.GKLS_minima.local_min[index])
        a = norm * norm + self.GKLS_minima.f[0] - self.GKLS_minima.f[index]
        rho = self.GKLS_minima.rho[index]
        norm = self.GKLS_norm(self.GKLS_minima.local_min[index], x)
        scal = 0.0
        for i in range(self.GKLS_dim):
            scal += (x[i] - self.GKLS_minima.local_min[index][i]) * (
                    self.GKLS_minima.local_min[0][i] - self.GKLS_minima.local_min[index][i])
        # Return the value of the cubic interpolation function *#
        res = np.double(0)
        res = (2.0 / rho / rho * scal / norm - 2.0 * a / rho / rho / rho) * norm * norm * norm + (
                1.0 - 4.0 * scal / norm / rho + 3.0 * a / rho / rho) * norm * norm + self.GKLS_minima.f[index]
        return res

    def GKLS_norm(self, x1, x2):
        i = 0
        norm = np.double(0)
        for i in range(self.GKLS_dim):
            norm += (x1[i] - x2[i]) * (x1[i] - x2[i])
        return np.sqrt(norm)

    def SetFunctionClass(self, _GKLStype, classDimension):
        self.GKLS_dim = classDimension
        self.mDimension = self.GKLS_dim
        self.GKLS_num_minima = 10
        if (self.mIsDomainMemeoryAllocated):
            self.GKLS_domain_free()

        if (_GKLStype == GKLSClass.Simple):
            if self.mDimension == 2:
                self.GKLS_global_dist = 0.9
                self.GKLS_global_radius = 0.2
            elif self.mDimension == 3:
                self.GKLS_global_dist = 0.66
                self.GKLS_global_radius = 0.2

            elif self.mDimension == 4:
                self.GKLS_global_dist = 0.66
                self.GKLS_global_radius = 0.2

            elif self.mDimension == 5:
                self.GKLS_global_dist = 0.66
                self.GKLS_global_radius = 0.3

            else:
                self.GKLS_global_dist = 0.66
                self.GKLS_global_radius = 0.3

        else:

            if self.mDimension == 2:
                self.GKLS_global_dist = 0.9
                self.GKLS_global_radius = 0.1

            elif self.mDimension == 3:
                self.GKLS_global_dist = 0.90
                self.GKLS_global_radius = 0.2

            elif self.mDimension == 4:
                self.GKLS_global_dist = 0.90
                self.GKLS_global_radius = 0.2

            elif self.mDimension == 5:
                self.GKLS_global_dist = 0.66
                self.GKLS_global_radius = 0.2

            else:
                self.GKLS_global_dist = 0.66
                self.GKLS_global_radius = 0.2

        self.GKLS_global_value = GKLSFunction.GKLS_GLOBAL_MIN_VALUE

    def SetFunctionNumber(self, number):
        self.mFunctionNumber = number
        if self.mIsGeneratorMemoryAllocated:
            self.GKLS_free()
        err_code = self.GKLS_arg_generate(number)

    def GKLS_arg_generate(self, nf):
        i = 0
        j = 0
        error = 0
        sin_phi = np.double(0)  # for generating of the global minimizer coordinates #
        # by using the generalized spherical coordinates #
        gap = self.GKLS_global_radius  # gap > 0 #
        # the minimal distance of any local minimizer to the attraction#
        # region of the global minimizer M(1); the value #
        # GKLS_global_radius is given by default and can be changed, #
        # but it should not be too small.  #

        # Check function number #
        if ((nf < 1) or (nf > 100)):
            return GKLSFunction.GKLS_FUNC_NUMBER_ERROR

        # Check parameters #
        error = self.GKLS_parameters_check()
        if (error != GKLSFunction.GKLS_OK):
            return error

        # Allocate memory #
        error = self.GKLS_alloc()
        if (error != GKLSFunction.GKLS_OK):
            return error

        # Set random seed #
        error = self.GKLS_initialize_rnd(self.GKLS_dim, self.GKLS_num_minima, nf)
        if (error != GKLSFunction.GKLS_OK):
            return error

        self.mRndGenerator.GenerateNextNumbers()  # ranf_array(rnd_num, NUM_RND); # get random sequence #
        self.rnd_counter = 0  # index of the random element from the sequence #
        # to be used as the next random number #

        # Set the paraboloid minimizer coordinates and #
        # the paraboloid minimum value #
        for i in range(self.GKLS_dim):
            self.GKLS_minima.local_min[0][i] = self.GKLS_domain_left[i] + self.rnd_num[self.rnd_counter] * (
                    self.GKLS_domain_right[i] - self.GKLS_domain_left[i])
            self.rnd_counter += 1
            if (self.rnd_counter == GKLSRandomGenerator.NUM_RND):
                self.mRndGenerator.GenerateNextNumbers()  # ranf_array(rnd_num, NUM_RND);
                self.rnd_counter = 0

        # for coordinates #
        self.GKLS_minima.f[0] = GKLSFunction.GKLS_PARABOLOID_MIN  # fix the paraboloid min value #

        # Generate the global minimizer using generalized spherical
        # coordinates#
        # with an arbitrary vector phi and the fixed radius GKLS_global_radius
        # #

        # First, generate an angle 0 <= phi(0) <= GKLSFunction.PI, and the coordinate x(0)#
        self.mRndGenerator.GenerateNextNumbers()  # ranf_array(rnd_num, NUM_RND);
        self.rnd_counter = 0

        self.GKLS_minima.local_min[1][0] = self.GKLS_minima.local_min[0][0] + self.GKLS_global_dist * np.cos(
            GKLSFunction.PI * self.rnd_num[self.rnd_counter])
        if ((self.GKLS_minima.local_min[1][0] > self.GKLS_domain_right[0] - GKLSFunction.GKLS_PRECISION) or (
                self.GKLS_minima.local_min[1][0] < self.GKLS_domain_left[0] + GKLSFunction.GKLS_PRECISION)):
            self.GKLS_minima.local_min[1][0] = self.GKLS_minima.local_min[0][0] - self.GKLS_global_dist * np.cos(
                GKLSFunction.PI * self.rnd_num[self.rnd_counter])
        sin_phi = np.sin(GKLSFunction.PI * self.rnd_num[self.rnd_counter])
        self.rnd_counter += 1

        # Generate the remaining angles 0<=phi(i)<=2*GKLSFunction.PI, and #
        # the coordinates x(i), i=1,...,GKLS_dim-2 (not last!) #
        for j in range(1, self.GKLS_dim - 1):
            self.GKLS_minima.local_min[1][j] = self.GKLS_minima.local_min[0][j] + self.GKLS_global_dist * np.cos(
                2.0 * GKLSFunction.PI * self.rnd_num[self.rnd_counter]) * sin_phi
            if ((self.GKLS_minima.local_min[1][j] > self.GKLS_domain_right[j] - GKLSFunction.GKLS_PRECISION) or (
                    self.GKLS_minima.local_min[1][j] < self.GKLS_domain_left[j] + GKLSFunction.GKLS_PRECISION)):
                self.GKLS_minima.local_min[1][j] = self.GKLS_minima.local_min[0][j] - self.GKLS_global_dist * np.cos(
                    2.0 * GKLSFunction.PI * self.rnd_num[self.rnd_counter]) * sin_phi
            sin_phi *= np.sin(2.0 * GKLSFunction.PI * self.rnd_num[self.rnd_counter])
            self.rnd_counter = self.rnd_counter + 1

        # Generate the last coordinate x(GKLS_dim-1) #
        self.GKLS_minima.local_min[1][self.GKLS_dim - 1] = \
            self.GKLS_minima.local_min[0][self.GKLS_dim - 1] + self.GKLS_global_dist * sin_phi
        if ((self.GKLS_minima.local_min[1][self.GKLS_dim - 1] >
             self.GKLS_domain_right[self.GKLS_dim - 1] - GKLSFunction.GKLS_PRECISION) or
                (self.GKLS_minima.local_min[1][self.GKLS_dim - 1] <
                 self.GKLS_domain_left[self.GKLS_dim - 1] + GKLSFunction.GKLS_PRECISION)):
            self.GKLS_minima.local_min[1][self.GKLS_dim - 1] = \
                self.GKLS_minima.local_min[0][self.GKLS_dim - 1] - self.GKLS_global_dist * sin_phi

        # Set the global minimum value #
        self.GKLS_minima.f[1] = self.GKLS_global_value

        # Set the weight coefficients w_rho(i) #
        for i in range(self.GKLS_num_minima):
            self.GKLS_minima.w_rho[i] = 0.99
            self.GKLS_minima.w_rho[1] = 1.0

        # Set the parameter delta for D2-type functions #
        # It is chosen randomly from (0,GKLS_DELTA_MAX_VALUE) #
        self.delta = GKLSFunction.GKLS_DELTA_MAX_VALUE * self.rnd_num[self.rnd_counter]
        self.rnd_counter += 1
        if (self.rnd_counter == GKLSRandomGenerator.NUM_RND):
            self.mRndGenerator.GenerateNextNumbers()  # ranf_array(rnd_num, NUM_RND);
            self.rnd_counter = 0

        # Choose randomly coordinates of local minimizers #
        # This procedure is repeated while the local minimizers #
        # coincide (external do...while); #
        # The internal cycle do..while serves to choose local #
        # minimizers in certain distance from the attraction #
        # region of the global minimizer M(i) #
        while (True):
            i = 2
            while (i < self.GKLS_num_minima):
                while (True):
                    self.mRndGenerator.GenerateNextNumbers()  # ranf_array(rnd_num, NUM_RND);
                    self.rnd_counter = 0

                    for j in range(self.GKLS_dim):
                        self.GKLS_minima.local_min[i][j] = self.GKLS_domain_left[j] + self.rnd_num[self.rnd_counter] * (
                                self.GKLS_domain_right[j] - self.GKLS_domain_left[j])
                        self.rnd_counter += 1
                        if (self.rnd_counter == GKLSRandomGenerator.NUM_RND):
                            self.mRndGenerator.GenerateNextNumbers()  # ranf_array(rnd_num, NUM_RND);
                            self.rnd_counter = 0

                    if not ((self.GKLS_global_radius + gap) -
                            self.GKLS_norm(self.GKLS_minima.local_min[i],
                                           self.GKLS_minima.local_min[1]) > GKLSFunction.GKLS_PRECISION):
                        break
                i += 1
            error = self.GKLS_coincidence_check()
            if not ((error == GKLSFunction.GKLS_PARABOLA_MIN_COINCIDENCE_ERROR) or
                    (error == GKLSFunction.GKLS_LOCAL_MIN_COINCIDENCE_ERROR)):
                break

        error = self.GKLS_set_basins()
        if (error == GKLSFunction.GKLS_OK):
            self.isArgSet = 1  # All the parameters are set #
        # and the user can Calculate a specific test function or #
        # its partial derivative by calling corresponding subroutines #

        return error

    def GKLS_parameters_check(self):

        i = 0
        min_side = np.double(0)
        tmp = np.double(0)

        if ((self.GKLS_dim <= 1) or (self.GKLS_dim >= GKLSRandomGenerator.NUM_RND)):
            return GKLSFunction.GKLS_DIM_ERROR  # problem dimension errors #
        if (self.GKLS_num_minima <= 1):  # number of local minima error #
            return GKLSFunction.GKLS_NUM_MINIMA_ERROR

        for i in range(self.GKLS_dim):
            if (self.GKLS_domain_left[i] >= self.GKLS_domain_right[i] - GKLSFunction.GKLS_PRECISION):
                return GKLSFunction.GKLS_BOUNDARY_ERROR  # the boundaries are erroneous #
        if (self.GKLS_global_value >= GKLSFunction.GKLS_PARABOLOID_MIN - GKLSFunction.GKLS_PRECISION):
            return GKLSFunction.GKLS_GLOBAL_MIN_VALUE_ERROR  # the global minimum value must #
        # be less than the paraboloid min #
        # Find min_side = min |b(i)-a(i)|, D=[a,b], and #
        # check the distance from paraboloid vertex to global minimizer #
        min_side = self.GKLS_domain_right[0] - self.GKLS_domain_left[0]
        for i in range(self.GKLS_dim):
            tmp = self.GKLS_domain_right[i] - self.GKLS_domain_left[i]
            if (tmp < min_side):
                min_side = tmp
        if ((self.GKLS_global_dist >= 0.5 * min_side - GKLSFunction.GKLS_PRECISION) or (
                self.GKLS_global_dist <= GKLSFunction.GKLS_PRECISION)):
            return GKLSFunction.GKLS_GLOBAL_DIST_ERROR  # global distance error #
        if ((self.GKLS_global_radius >= 0.5 * self.GKLS_global_dist + GKLSFunction.GKLS_PRECISION) or (
                self.GKLS_global_radius <= GKLSFunction.GKLS_PRECISION)):
            return GKLSFunction.GKLS_GLOBAL_RADIUS_ERROR  # global minimizer attr.  radius error #

        return GKLSFunction.GKLS_OK  # no errors #

    def GKLS_alloc(self):
        i = 0

        if ((self.GKLS_dim <= 1) or (self.GKLS_dim >= GKLSRandomGenerator.NUM_RND)):
            return GKLSFunction.GKLS_DIM_ERROR  # problem dimension error #
        if (self.GKLS_num_minima <= 1):
            return GKLSFunction.GKLS_NUM_MINIMA_ERROR  # erroneous number of local minima #
        self.GKLS_minima.local_min = np.zeros((self.GKLS_num_minima, self.GKLS_dim), dtype=np.double)

        self.GKLS_minima.w_rho = np.zeros(self.GKLS_num_minima, dtype=np.double)
        self.GKLS_minima.peak = np.zeros(self.GKLS_num_minima, dtype=np.double)
        self.GKLS_minima.rho = np.zeros(self.GKLS_num_minima, dtype=np.double)
        self.GKLS_minima.f = np.zeros(self.GKLS_num_minima, dtype=np.double)
        self.GKLS_glob.gm_index = np.zeros(self.GKLS_num_minima, dtype=int)
        self.GKLS_glob.num_global_minima = 0
        self.mIsGeneratorMemoryAllocated = True
        return GKLSFunction.GKLS_OK  # no errors #

    def GKLS_initialize_rnd(self, dim, nmin, nf):

        seed = 0
        # seed number between 0 and 2^30-3 = 1,073,741,821#

        seed = (nf - 1) + (nmin - 1) * 100 + dim * 1000000
        # If big values of nmin and dim are required, #
        # one must check wether seed <= 1073741821 #

        self.mRndGenerator.Initialize(seed, self.rnd_num, self.rand_condition)  # ranf_start(seed);

        return GKLSFunction.GKLS_OK

    def GKLS_coincidence_check(self):

        i = 0
        j = 0

        # Check wether some local minimizer coincides with the paraboloid minimizer #
        for i in range(2, self.GKLS_num_minima):
            if (self.GKLS_norm(self.GKLS_minima.local_min[i],
                               self.GKLS_minima.local_min[0]) < GKLSFunction.GKLS_PRECISION):
                return GKLSFunction.GKLS_PARABOLA_MIN_COINCIDENCE_ERROR

        # Check wether there is a pair of identical local minimizers #
        for i in range(1, self.GKLS_num_minima - 1):
            for i in range(i + 1, self.GKLS_num_minima):
                if (self.GKLS_norm(self.GKLS_minima.local_min[i],
                                   self.GKLS_minima.local_min[j]) < GKLSFunction.GKLS_PRECISION):
                    return GKLSFunction.GKLS_LOCAL_MIN_COINCIDENCE_ERROR
        return GKLSFunction.GKLS_OK

    def GKLS_set_basins(self):

        i = 0
        j = 0
        temp_min = np.double(0)  # temporary  #
        temp_d1 = np.double(0)
        temp_d2 = np.double(0)  # variables  #
        dist = np.double(0)  # for finding the distance between two minimizers #

        # **************************************************************************#
        # First, set the radii rho(i) of the attraction regions: these values are  #
        # defined in such a way that attraction regions are as large as possible   #
        # and do not overlap; it is not required that the attraction region of each#
        # local minimizer be entirely contained in D. The values found in such     #
        # a manner are corrected then by the weight coefficients w(i)              #
        # **************************************************************************#

        # Calculate dist(i) - the minimal distance from the minimizer i to         #
        #                     the other minimizers.                                #
        # Set the initial value of rho(i) as rho(i) = dist(i)/2: so the attraction #
        # regions do not overlap                                                   #
        for i in range(self.GKLS_num_minima):
            temp_min = GKLSFunction.GKLS_MAX_VALUE
            for j in range(self.GKLS_num_minima):
                if (i != j):
                    temp_d1 = self.GKLS_norm(self.GKLS_minima.local_min[i], self.GKLS_minima.local_min[j])
                    if (temp_d1 < temp_min):
                        temp_min = temp_d1
            dist = temp_min / 2.0
            self.GKLS_minima.rho[i] = dist

        # Since the radius of the attraction region of the global minimizer            #
        # is fixed by the user, the generator adjusts the radii of the attraction      #
        # regions, eventually overlapping with the attraction region of the global     #
        # minimizer. To do this, it checks whether the attraction region radius        #
        # of each local minimizer exceeds the distance between this minimizer          #
        # and the attraction region of the global minimizer.                           #
        # If such a situation is verified the generator decreases the attraction       #
        # region radius of the local minimizer setting it equal to he distance between #
        # the local minimizer and the attraction region of the global minimizer.       #
        # Note that the radius of the attraction region of the global minimizer can    #
        # not be greater than one half of the distance (defined by the user) between   #
        # the global minimizer and the paraboloid vertex. So, the initially defined    #
        # attraction regions of the global minimizer and the paraboloid vertex do not  #
        # overlap even when the global minimizer is the closest minimizer to the       #
        # paraboloid vertex.                                                           #
        self.GKLS_minima.rho[1] = self.GKLS_global_radius
        for i in range(2, self.GKLS_num_minima):
            dist = (self.GKLS_norm(self.GKLS_minima.local_min[i], self.GKLS_minima.local_min[
                1]) - self.GKLS_global_radius - GKLSFunction.GKLS_PRECISION)
            if (dist < self.GKLS_minima.rho[i]):
                self.GKLS_minima.rho[i] = dist

        # Try to expand the attraction regions of local minimizers until they      #
        # do not overlap                                                           #
        for i in range(self.GKLS_num_minima):

            if (i != 1):  # The radius of the attr. region of the global min is fixed  #
                # rho(i) := max {rho(i),min[||M(i)-M(j)|| - rho(j): i !=j]},      #
                temp_min = GKLSFunction.GKLS_MAX_VALUE
                for j in range(self.GKLS_num_minima):
                    if (i != j):
                        temp_d1 = self.GKLS_norm(self.GKLS_minima.local_min[i], self.GKLS_minima.local_min[j]) - \
                                  self.GKLS_minima.rho[j]
                        if (temp_d1 < temp_min):
                            temp_min = temp_d1

                # Increase the radius rho(i) if it is possible #
                if (temp_min > self.GKLS_minima.rho[i] + GKLSFunction.GKLS_PRECISION):
                    self.GKLS_minima.rho[i] = temp_min

        # Correct the radii by weight coefficients w(i)                    #
        # The weight coefficients can be chosen randomly;                  #
        # here they are defined by default as:                             #
        #    w(i) = 0.99, i != 1 , and w(1) = 1.0 (global min index = 1)   #
        for i in range(self.GKLS_num_minima):
            self.GKLS_minima.rho[i] = self.GKLS_minima.w_rho[i] * self.GKLS_minima.rho[i]

        # *******************************************************************#
        # Set the local minima values f(i) of test functions as follows:    #
        #   f(i) = cond_min(i) - peak(i), i != 1 (global min index = 1)     #
        #   f(0) = GKLS_PARABOLOID_MIN, f(1) = GKLS_GLOBAL_MIN_VALUE,       #
        # where cond_min(i) is the paraboloid minimum value at the boundary #
        # B={||x-M(i)||=rho(i)} of the attraction region of the local       #
        # minimizer M(i), i.e.                                              #
        #  cond_min(i) =                                                    #
        #  = paraboloid g() value at (M(i)+rho(i)*(T-M(i))/norm(T-M)) =     #
        #  = (rho(i) - norm(T-M(i)))^2 + t,                                 #
        #  g(x) = ||x-T||^2 + t, x in D of R^GKLS_dim                       #
        #                                                                   #
        #  The values of peak(i) are chosen randomly from an interval       #
        # (0, 2*rho(i), so that the values f(i) depend on radii rho(i) of   #
        # the attraction regions, 2<=i<GKLS_dim.                            #
        #  The condition f(x*)=f(1) <= f(i) must be satisfied               #
        # *******************************************************************#
        # Fix to 0 the peak(i) values of the paraboloid and the global min  #
        self.GKLS_minima.peak[0] = 0.0
        self.GKLS_minima.peak[1] = 0.0
        for i in range(2, self.GKLS_num_minima):
            # Set values peak(i), i>= 2 #
            # Note that peak(i) is such that the function value f(i) is smaller#
            # than min(GKLS_GLOBAL_MIN_VALUE, 2*rho(i))                        #
            temp_d1 = self.GKLS_norm(self.GKLS_minima.local_min[0], self.GKLS_minima.local_min[i])

            # the conditional minimum at the boundary#
            temp_min = (self.GKLS_minima.rho[i] - temp_d1) * \
                       (self.GKLS_minima.rho[i] - temp_d1) + self.GKLS_minima.f[0]

            temp_d1 = (1.0 + self.rnd_num[self.rnd_counter]) * self.GKLS_minima.rho[i]
            temp_d2 = self.rnd_num[self.rnd_counter] * (temp_min - self.GKLS_global_value)
            # temp_d1 := min(temp_d1, temp_d2) #
            if temp_d2 < temp_d1:
                temp_d1 = temp_d2
            self.GKLS_minima.peak[i] = temp_d1

            self.rnd_counter += 1
            if self.rnd_counter == GKLSRandomGenerator.NUM_RND:
                self.mRndGenerator.GenerateNextNumbers()  # ranf_array(self.rnd_num, NUM_RND);
                self.rnd_counter = 0

            self.GKLS_minima.f[i] = temp_min - self.GKLS_minima.peak[i]

        # *******************************************************************#
        # Find all possible global minimizers and                           #
        # create a list of their indices among all the minimizers           #
        # Note that the paraboloid minimum can not be the global one because#
        # the global optimum value is set to be less than the paraboloid    #
        # minimum value                                                     #
        # *******************************************************************#
        self.GKLS_glob.num_global_minima = 0
        for i in range(self.GKLS_num_minima):
            if ((self.GKLS_minima.f[i] >= self.GKLS_global_value - GKLSFunction.GKLS_PRECISION) and (
                    self.GKLS_minima.f[i] <= self.GKLS_global_value + GKLSFunction.GKLS_PRECISION)):

                self.GKLS_glob.gm_index[self.GKLS_glob.num_global_minima] = i
                self.GKLS_glob.num_global_minima += 1
                # The first GKLS_glob.num_global_minima elements of the list    #
                # contain the indices of the global minimizers                  #

            else:
                self.GKLS_glob.gm_index[self.GKLS_num_minima - 1 - i + self.GKLS_glob.num_global_minima] = i
        # The remaining elements of the list                            #
        # contain the indices of local (non global) minimizers          #

        if (self.GKLS_glob.num_global_minima == 0):  # erroneous case:       #
            return GKLSFunction.GKLS_FLOATING_POINT_ERROR  # some programmer's error #

        return GKLSFunction.GKLS_OK

    def GKLS_domain_alloc(self):

        i = 0

        if ((self.GKLS_dim <= 1) or (self.GKLS_dim >= GKLSRandomGenerator.NUM_RND)):
            return GKLSFunction.GKLS_DIM_ERROR  # problem dimension error */
        self.GKLS_domain_left = np.zeros(self.GKLS_dim, dtype=np.double)
        self.GKLS_domain_right = np.zeros(self.GKLS_dim, dtype=np.double)

        for i in range(self.GKLS_dim):
            self.GKLS_domain_left[i] = -1.0
            self.GKLS_domain_right[i] = 1.0

        self.mIsDomainMemeoryAllocated = True
        return GKLSFunction.GKLS_OK  # no errors */

    def SetDimension(self, value):
        # assert(value > 1 && value <= NUM_RND);
        self.mDimension = value
        self.GKLS_dim = value
        if self.mIsDomainMemeoryAllocated:
            self.GKLS_domain_free()
        self.GKLS_domain_alloc()

    def GetOptimumValue(self):
        return self.GKLS_global_value

    def GetOptimumPoint(self):
        if (self.isArgSet == 1):
            argmin = self.GKLS_minima.local_min[self.GKLS_glob.gm_index[0]]
            return argmin
        return -1
