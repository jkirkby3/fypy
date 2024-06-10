import numpy as np

# np.set_printoptions(precision=4)

from fypy.model.FourierModel import FourierModel

from fypy.pricing.fourier.StochVol.StochVolParams import (
    Toeplitz,
    GridParamsGeneric,
    NumericalParams,
    ExponentialMat,
    PayoffConstants,
    AddJumpsCharacteristics,
)
from fypy.pricing.fourier.StochVol.StochVolPricer import NonRecursivePriceGeneric


class GridParams(GridParamsGeneric):
    def __init__(self, W: float, S0: float, N: int, alpha: float, T: float, M: int):
        super().__init__(W, S0, N, alpha, T, M)
        self.init_variables()
        self.conditional_variables()
        self.update_and_create_variables()

    def init_variables(self, **kwargs):
        self.dx = self._get_dx()
        self.a = self._get_a()
        self.dxtil = self._get_dxtil()
        self.K = self._get_K()
        self.nnot = self._get_nnot()
        self.xmin = self._get_xmin()
        self.nbar = self._get_nbar()
        return

    def conditional_variables(self):
        if abs(self.lws) < self.dxtil:
            self.dx = self.dxtil
        elif self.lws < 0:
            self.dx = self.lws / (1 + self.nbar - self.K / 2)
            self.nbar += 1
        elif self.lws > 0:
            self.dx = self.lws / (self.nbar - self.K / 2)
        return

    def update_and_create_variables(self):
        self.pf_cons = PayoffConstants(self.dx, self.gauss_quad.b4, self.gauss_quad.b3)
        self.a = 1 / self.dx
        self.xmin = (1 - self.K / 2) * self.dx
        self.rho = self._get_rho()
        self.zeta = self._get_zeta()
        self.dxi = self._get_dxi()
        self.xi = self._get_xi()
        self.gs = self._get_gs()
        self.thetM = self._get_thetM()
        return


class ProjBermudanPricer_SV:
    def __init__(self, model: FourierModel, N: int = 2**11):
        self._init_constants(N)
        self.model = AddJumpsCharacteristics(model).get_model()

    def _init_constants(self, N: int):
        self.N = N

    # Main funtion: pricing method
    def price(self, T: float, W: int, S0: float, M: int, is_call: bool) -> float:
        self._check_call_put(is_call=is_call)
        num_params = NumericalParams()
        grid = GridParams(W, S0, self.N, alpha=num_params.alpha, T=T, M=M)
        exp_mat = ExponentialMat(grid, self.model, num_params, T)
        toep = Toeplitz(grid, self.model)
        recursion = RecursivePricer(exp_mat, toep, grid, num_params)
        price = recursion.recursive_price()
        return price

    # need to initialize ProjPricer
    def _beta_computation(self, exp_mat: np.ndarray):
        raise NotImplementedError

    # check that contract asked is implemented (NOTE: to move in a proper class ?)
    def _check_call_put(self, is_call: bool):
        match is_call:
            case True:
                raise NotImplementedError("Only put pricing is implemented so far.")
            case False:
                pass


class RecursivePricer(NonRecursivePriceGeneric):
    def __init__(
        self,
        mat: ExponentialMat,
        toep: Toeplitz,
        grid: GridParams,
        num_params: NumericalParams,
    ):
        super().__init__(mat, toep, grid, num_params)
        return

    def _get_val(self, k0: int):
        nnot = self.grid.nnot
        val1 = np.max([self.cont[nnot - 1, k0 - 1], self.grid.gs[nnot - 1]])
        val2 = np.max([self.cont[nnot - 1, k0], self.grid.gs[nnot - 1]])
        return val1, val2

    def _init_variables(self):
        self._thet = self._init_thet()
        self.cont = self._create_cont()
        self._kstr = np.full(self.num_params.m0, self.grid.nbar - 1, dtype="int")
        return

    def _init_thet(self):
        return np.zeros((self.grid.K, self.num_params.m0))

    def _vec_for_thet_update(self):
        return self.grid.thetM

    def _update_thet(self, j: int) -> None:
        self._update_thet_constants(j)
        self._update_thet_sub(j)

    def _update_thet_constants(self, j):
        self._while(j)
        self._update_kstr(j)
        self.rho = self._xstrs - self._xkstr
        self._update_zeta()
        # NOTE: Why don't move all of these constant in a class depending only self.grid, self.gauss_quad and _xstrs skstr
        self._update_ed()
        self._update_dbar()
        self._update_d()
        self.idx_j = self._kstr[j]
        self.ck4 = self.cont[self.idx_j + 2, j]
        return

    def _update_d(self):
        self.d0 = self._get_d0()
        self.d1 = self._get_d1()

    def _update_thet_sub(self, j):
        idx_j = self.idx_j
        self._thet[:idx_j, j] = self.grid.thetM[:idx_j]
        self._thet[self.grid.K - 1, j] = self._update_thet_sub1(j)
        self._thet[idx_j + 1, j] = self._update_thet_sub2(
            self.dbar1, self.zeta, self.d1
        )
        self._thet[idx_j + 2 : (self.grid.K - 1), j] = self._update_thet_sub3(j, idx_j)
        self._thet[idx_j, j] = self._update_thet_sub4(self.dbar0, self.zeta, self.d0)

    def _update_zeta(self):
        self.zeta = self.grid.a * self.rho
        self.zeta_minus = self.zeta * self.grid.gauss_quad.q_minus
        self.zeta_plus = self.zeta * self.grid.gauss_quad.q_plus

    def _update_ed(self):
        self.ed1 = np.exp(self.rho * self.grid.gauss_quad.q_minus)
        self.ed2 = np.exp(0.5 * self.rho)
        self.ed3 = np.exp(self.rho * self.grid.gauss_quad.q_plus)

    def _update_dbar(self):
        self.dbar1 = self.zeta**2 / 2
        self.dbar0 = self.zeta - self.dbar1

    def _get_d1(self):
        return (
            np.exp(-self.grid.dx)
            * self.zeta
            * (
                5 * (self.zeta_minus * self.ed1 + self.zeta_plus * self.ed3)
                + 4 * self.zeta * self.ed2
            )
            / 18
        )

    def _get_d0(self):
        return (
            self.zeta
            * (
                5 * ((1 - self.zeta_minus) * self.ed1 + (1 - self.zeta_plus) * self.ed3)
                + 4 * (2 - self.zeta) * self.ed2
            )
            / 18
        )

    def _update_thet_sub4(self, dbar0: float, zeta: float, d0: float):
        return (
            self.grid.W * (0.5 + dbar0)
            - self.grid.S0 * np.exp(self._xkstr) * (self.grid.pf_cons.varthet_m10 + d0)
            + zeta**4 / 8 * (self._ck1 - 2 * self._ck2 + self._ck3)
            + zeta**3 / 3 * (self._ck2 - self._ck1)
            + zeta**2 / 4 * (self._ck1 + 2 * self._ck2 - self._ck3)
            - zeta * self._ck2
            - self._ck1 / 24
            + 5 / 12 * self._ck2
            + self._ck3 / 8
        )

    def _update_thet_sub3(self, j: int, idx_j: np.ndarray | int) -> np.ndarray:
        return (
            self.cont[idx_j + 1 : self.grid.K - 2, j]
            + 10 * self.cont[idx_j + 2 : self.grid.K - 1, j]
            + self.cont[idx_j + 3 : self.grid.K, j]
        ) / 12

    def _update_thet_sub2(self, dbar1: float, zeta: float, d1: float) -> float:
        return (
            self.grid.W * dbar1
            - self.grid.S0 * np.exp(self._xkstr + self.grid.dx) * d1
            + zeta**4 / 8 * (-self._ck2 + 2 * self._ck3 - self.ck4)
            + zeta**3 / 6 * (3 * self._ck2 - 4 * self._ck3 + self.ck4)
            - 0.5 * zeta**2 * self._ck2
            + (self._ck2 + 10 * self._ck3 + self.ck4) / 12
        )

    def _update_thet_sub1(self, j: int) -> float:
        return (
            13 * self.cont[self.grid.K - 1, j]
            + 15 * self.cont[self.grid.K - 2, j]
            - 5 * self.cont[self.grid.K - 3, j]
            + self.cont[self.grid.K - 4, j]
        ) / 48

    def _while(self, j: int) -> None:
        idx = self._kstr[j]
        while idx > 1 and (self.cont[idx, j] > self.grid.gs[idx]):
            idx -= 1
            self._kstr[j] = idx
        return

    def _update_kstr(self, j: int) -> None:
        idx_j = self._kstr[j]
        if self._kstr[j] >= 1:
            self._update_ck(j, idx_j)
            self._update_gk(idx_j)
            self._update_temp()
            self._update_x(idx_j)

        else:
            self._update_x_other_case(j)
        return

    def _update_x_other_case(self, j):
        self._kstr[j] = 1
        self._xstrs = self.grid.xmin
        self._xkstr = self.grid.xmin

    def _update_x(self, idx_j):
        self._xkstr = self.grid.xmin + (idx_j) * self.grid.dx
        self._xstrs = (
            (self._xkstr + self.grid.dx) * self.tmp1 - self._xkstr * self.tmp2
        ) / (self.tmp1 - self.tmp2)

    def _update_temp(self):
        self.tmp1 = self._ck2 - self.gk2
        self.tmp2 = self._ck3 - self.gk3

    def _update_gk(self, idx_j):
        self.gk2 = self.grid.gs[idx_j]
        self.gk3 = self.grid.gs[idx_j + 1]

    def _update_ck(self, j, idx_j):
        self._ck1 = self.cont[idx_j - 1, j]
        self._ck2 = self.cont[idx_j, j]
        self._ck3 = self.cont[idx_j + 1, j]
