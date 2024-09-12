import numpy as np

# np.set_printoptions(precision=4)

from fypy.model.FourierModel import FourierModel


from fypy.pricing.fourier.StochVol.StochVolParams import (
    GridParamsGeneric,
    NumericalParams,
    ExponentialMat,
    AlphaRecursiveReturn,
    TYPES,
)

from fypy.pricing.fourier.StochVol.StochVolPricer import RecursiveReturnPricer


class GridParams(GridParamsGeneric):
    def __init__(
        self,
        W: float,
        S0: float,
        N: int,
        alpha: float,
        T: float,
        M: int,
        C: float,
        F: float,
        contract: int,
    ):
        super().__init__(W, S0, N, alpha, T, M)
        self.init_variables(alpha, C, F)
        self.conditional_variables(contract, C, F)
        self.update_and_create_variables()

    def init_variables(self, alpha: float, C: float, F: float):
        self.alpha = alpha
        self.dx = 2 * self.alpha / (self.N - 1)
        self.a = 1 / self.dx
        self.dt = self.T / self.M
        self.xmin = (1 - self.N / 2) * self.dx
        self.lc = np.log(1 + C)
        self.lf = np.log(1 + F)
        self.klc = int(np.floor(self.a * (self.lc - self.xmin)) + 1)
        xklc = self.xmin + (self.klc - 1) * self.dx
        self.xmin = self.xmin + (self.lc - xklc)
        self.klf = int(np.floor(self.a * (self.lf - self.xmin)) + 1)
        return

    def conditional_variables(self, contract: int, C: float, F: float):
        match contract:
            case 2 | 3 | 4:
                if self.klc != self.klf:
                    self.dx = (self.lc - self.lf) / (self.klc - self.klf)
                    self.a = 1 / self.dx
                    self.xmin = self.lf - (self.klf - 1) * self.dx

                self.hlocalCF = (
                    lambda x: F * (x <= self.lf)
                    + (np.exp(x) - 1) * (x < self.lc) * (x > self.lf)
                    + C * (x >= self.lc)
                )
            case _:
                raise NotImplementedError
        return

    def update_and_create_variables(self):
        self.A = 32 * self.a**4
        self.C_an = self.A / self.N
        self.dxi = 2 * np.pi * self.a / self.N
        self.xi = self.dxi * np.arange(self.N)
        return


class RecursivePricer(RecursiveReturnPricer):
    def __init__(
        self,
        model: FourierModel,
        mat: ExponentialMat,
        grid: GridParamsGeneric,
        num_params: NumericalParams,
    ):
        super().__init__(model, grid, num_params, mat)

    def _set_left_and_NMM(self, contract: int):
        match contract:
            case 2 | 3 | 4:
                self.leftGridPoint = self.grid.lf - self.grid.dx
                # this is the left bound of gaussian quadrature grid
                self.NNM = self.grid.klc - self.grid.klf + 3
                # this is N_Psi
            case 1 | 5:
                raise NotImplementedError
            case _:
                raise NotImplementedError

    def _init_phi(self):
        self.phi_old = np.zeros((self.grid.N - 1, self.num_params.m0), dtype="complex_")
        self.phi_y_new = np.zeros(
            (self.grid.N - 1, self.num_params.m0), dtype="complex_"
        )
        return

    def _get_phi_new(self, C: float, F: float):
        self._set_expFC(C, F)
        self._build_phi_new_iter()
        return

    def _build_phi_new_iter(self):
        for j in range(self.num_params.m0):
            self._set_column_phi_old(j)
            beta_temp = np.real(
                np.fft.fft(np.hstack((1 / self.grid.A, self.phi_old[:, j] * self.hvec)))
            )
            sumBetaLeft = self._get_sum_beta_left(beta_temp)
            sumBetaRight = self._get_sum_beta_right(beta_temp, sumBetaLeft)
            self._set_column_phi_new(j, sumBetaLeft, sumBetaRight, beta_temp)

    def _set_expFC(self, C, F):
        self.expFxi = np.exp(1j * F * self.grid.xi)
        self.expCxi = np.exp(1j * C * self.grid.xi)

    def _set_column_phi_old(self, j):
        for n in range(self.grid.N - 1):
            self.phi_old[n, j] = self.transit_mat[:, j, n + 1].sum()

    def _set_column_phi_new(self, j, sumBetaLeft, sumBetaRight, beta_temp):
        col = self.psi @ beta_temp[self.grid.klf - 2 : self.grid.klc + 1]
        self.phi_y_new[:, j] = (
            col + self.expFxi * sumBetaLeft + self.expCxi * sumBetaRight
        )

    def _get_sum_beta_right(self, beta_temp, sumBetaLeft):
        return (
            1
            - sumBetaLeft
            - self.grid.C_an * beta_temp[self.grid.klf - 2 : self.grid.klc + 1].sum()
        )

    def _get_sum_beta_left(self, beta_temp):
        sumBetaLeft = self.grid.C_an * beta_temp[: self.grid.klf - 2].sum()
        return sumBetaLeft

    def _get_PhiY(self, C: float, F: float, contract: int):
        self._PSI_computation(contract, self.grid.a)
        self.transit_mat = self.exp_mat.get_exp_tensor()
        self.zeta = self._get_zeta_vec(self.grid.a, self.grid.xi)
        self.grid.xi = self.grid.xi[1:]
        self.hvec = np.exp(-1j * self.grid.xmin * self.grid.xi) * self.zeta
        self._init_phi()

        match contract:
            case 2 | 3 | 4:
                self._get_phi_new(C, F)
                self._get_phi()
            case _:
                raise NotImplementedError

        return

    def _get_phi(self):
        self.phi_y = np.zeros(
            (self.num_params.m0, self.num_params.m0, self.grid.N - 1), dtype="complex_"
        )
        xiBigF = self.expFxi
        xiBigC = self.expCxi

        if self.grid.M > 1:
            for j in range(self.num_params.m0):
                for k in range(self.num_params.m0):
                    self._build_line_phi(xiBigF, xiBigC, j, k)

    def _build_line_phi(self, xiBigF, xiBigC, j, k):
        grand = self.hvec * self.transit_mat[k, j, 1:]
        beta_temp = np.real(
            np.fft.fft(np.hstack((self.transit_mat[k, j, 0] / self.grid.A, grand)))
        )
        sum_left, sum_right = self._get_sums(beta_temp)
        self._compute_line_y(xiBigF, xiBigC, j, k, beta_temp, sum_left, sum_right)

    def _compute_line_y(self, xiBigF, xiBigC, j, k, beta_temp, sum_left, sum_right):
        line = self.psi @ beta_temp[self.grid.klf - 2 : self.grid.klc + 1]
        self.phi_y[j, k, :] = line + xiBigF * sum_left + xiBigC * sum_right
        return

    def _get_sums(self, beta_temp):
        sum_left = self.grid.C_an * beta_temp[: self.grid.klf - 2].sum()
        sum_right = self.grid.C_an * beta_temp[self.grid.klc + 1 :].sum()
        return sum_left, sum_right

    def _refining(self, contract: int, CG: float, FG: float):
        match contract:
            case 3:
                dx = self.grid.dx
                CmF = CG - FG
                ymin = FG - dx
                self.kc = np.floor(self.grid.a * (CG - ymin)) + 1
                z, z2, z3, z4, z5 = self._get_z(CG, dx, ymin)
                self._refine_thet(dx, CmF, z, z2, z3, z4, z5)
                k = int(self.kc + 2)
                self.theta[k:] = CmF
                if k > self.grid.N / 2:
                    raise ValueError
                self.hvec = np.exp(-1j * ymin * self.grid.xi) * self.zeta
            case _:
                raise NotImplementedError
        return

    def _refine_thet(self, dx, CmF, z, z2, z3, z4, z5):
        self._refine_thet_sub1(dx)
        self._refine_thet_sub2(dx, CmF, z, z2, z3, z4, z5)
        self._refine_thet_sub3(dx, CmF, z, z2, z3, z4, z5)
        self._refine_thet_sub4(dx, CmF, z, z2, z3, z4, z5)
        self._refine_thet_sub5(dx, CmF, z4, z5)

    def _get_z(self, CG, dx, ymin):
        z = self.grid.a * (CG - (ymin + (self.kc - 1) * dx))
        z2 = z**2
        z3 = z * z2
        z4 = z * z3
        z5 = z * z4
        return z, z2, z3, z4, z5

    def _refine_thet_sub1(self, dx):
        kc = self.kc
        self.theta = np.zeros(int(self.grid.N / 2))
        self.theta[0] = dx / 120
        self.theta[1] = dx * 7 / 30
        self.theta[2] = dx * 121 / 120
        self.theta[3 : int(kc - 2)] = dx * np.arange(2, int(kc - 3))

    def _refine_thet_sub5(self, dx, CmF, z4, z5):
        k = int(self.kc + 2)
        self.theta[k - 1] = dx * (z5 / 30 + (k - 4) * z4 / 24) + CmF * (1 - z4 / 24)
        # self.theta[k : int(self.grid.N / 2)] = CmF

    def _refine_thet_sub4(self, dx, CmF, z, z2, z3, z4, z5):
        k = int(self.kc + 1)
        self.theta[k - 1] = dx * (
            k * (-z4 / 8 + z3 / 6 + z2 / 4 + z / 6 + 1 / 24)
            - z5 / 10
            + z4 / 2
            - z3 / 3
            - 2 * z2 / 3
            - z / 2
            - 2 / 15
        ) + CmF * (0.5 + 1 / 24 * (3 * z4 - 4 * z3 - 6 * z2 - 4 * z + 11))

    def _refine_thet_sub3(self, dx, CmF, z, z2, z3, z4, z5):
        k = int(self.kc)
        self.theta[k - 1] = dx * (
            k * (z4 / 8 - z3 / 3 + 2 * z / 3 + 0.5)
            + z5 / 10
            - z4 / 2
            + 2 * z3 / 3
            + z2 / 3
            - 4 * z / 3
            - 37 / 30
        ) + CmF * (-z4 / 8 + z3 / 3 - 2 * z / 3 + 1 / 2)

    def _refine_thet_sub2(self, dx, CmF, z, z2, z3, z4, z5):
        k = int(self.kc - 1)
        self.theta[k - 1] = (
            dx
            * (
                k * (-z4 / 24 + z3 / 6 - z2 / 4 + z / 6 + 23 / 24)
                - z5 / 30
                + z4 / 6
                - z3 / 3
                + z2 / 3
                - z / 6
                - 59 / 30
            )
            + CmF * (z - 1) ** 4 / 24
        )


class ProjCliquetPricer_SV:
    def __init__(self, model: FourierModel, N: int = 2**10, L: float = 14):
        self.L = L
        self.N = N
        self.model = model

    def _beta_computation(self, k: int):
        beta = np.real(
            np.fft.fft(
                np.hstack(
                    (
                        1 / self.grid.A,
                        self.recursive_pricer.hvec
                        * self.recursive_pricer.phi_y_new[:, k],
                    )
                )
            )
        )
        return beta

    def price(
        self,
        T: float,
        M: int,
        W: float,
        S0: float,
        C: float,
        CG: float,
        F: float,
        FG: float,
        contract: int,
    ):

        self._init_constants(W, S0, T, M, contract, C, F)
        self.recursive_pricer._recursion()
        self.recursive_pricer._refining(contract, CG, FG)
        price = self._interpolation(contract, FG, W)
        return price

    def _init_constants(
        self, K: float, S0: float, T: float, M: float, contract: int, C: float, F: float
    ):
        self._init_classes(K, S0, T, M, contract, C, F)
        self.recursive_pricer._get_PhiY(C, F, contract)
        return

    def _init_classes(self, K, S0, T, M, contract, C, F):
        alpha = AlphaRecursiveReturn(self.model, T, self.L)()
        self.num_params = NumericalParams()
        self.grid = GridParams(K, S0, self.N, alpha, T, M, C, F, contract)
        self.exp_mat = ExponentialMat(self.grid, self.model, self.num_params, T)
        # self.recursive_pricer = RecursivePricer(
        #     self.model, self.grid, self.num_params, self.exp_mat
        # )
        self.recursive_pricer = RecursivePricer(
            self.model, self.exp_mat, self.grid, self.num_params
        )

    def _interpolation(self, contract: int, FG: float, W: float):
        if isinstance(self.model, TYPES.Hes_base):
            k0 = self.recursive_pricer._get_k0()
            disc = self.model.discountCurve.discount_T(self.grid.T)
            v0 = self.model.v_0
            v = self.exp_mat.get_v()
            match contract:
                case 3:
                    val1 = self._get_val(FG, W, k0 - 1, disc)
                    val2 = self._get_val(FG, W, k0, disc)
                    price = val1 + (val2 - val1) * (v0 - v[k0 - 1]) / (
                        v[k0] - v[k0 - 1]
                    )
                    return price
                case _:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    def _get_val(self, FG, W, k, disc):
        beta_temp = self._beta_computation(k)
        val1 = (
            W
            * disc
            * (
                FG
                + self.grid.C_an
                * self.recursive_pricer.theta
                @ beta_temp[: int(self.grid.N / 2)]
            )
        )

        return val1
