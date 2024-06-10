import numpy as np


from fypy.model.FourierModel import FourierModel


from fypy.pricing.fourier.StochVol.StochVolParams import (
    GridParamsGeneric,
    NumericalParams,
    ExponentialMat,
    Toeplitz,
    PayoffConstants,
)


from fypy.pricing.fourier.StochVol.StochVolPricer import NonRecursivePriceGeneric


class GridParams(GridParamsGeneric):
    def __init__(
        self,
        W: float,
        S0: float,
        N: int,
        H: int,
        alpha: float,
        T: float,
        M: int,
        down: bool,
    ):
        super().__init__(W=W, S0=S0, N=N, alpha=alpha, T=T, M=M, H=H, down=down)
        self.init_variables()
        self.conditional_variables()
        self.update_and_create_variables()
        return

    def init_variables(self, **kwargs):
        self.dx = self._get_dx()
        self.a = self._get_a()
        self.dxtil = self._get_dxtil()
        self.K = self._get_K()
        self.nnot = self._get_nnot()
        self.xmin = self._get_xmin()
        self.nbar = self._get_nbar()
        self.pf_cons = PayoffConstants(self.dx, self.gauss_quad.b4, self.gauss_quad.b3)
        return

    def conditional_variables(self):
        if self.H is not None:
            if self.down:
                self.xmin = np.log(self.H / self.S0)
                self.nnot = int(1 - self.xmin * self.a)
                self.dx = self.xmin / (1 - self.nnot)
                self.a = 1 / self.dx
            else:
                u = np.log(self.H / self.S0)
                self.lws = np.log(self.W / self.S0)
                self.nnot = int(self.K - self.a * u)
                self.dx = u / (self.K - self.nnot)
                self.a = 1 / self.dx
                self.xmin = u - (self.K - 1) * self.dx
        else:
            raise ValueError("H should be initialized")
        return

    def update_and_create_variables(self):
        self.nbar = self._get_nbar()
        self.rho = self._get_rho()
        self.zeta = self._get_zeta()
        self.dxi = self._get_dxi()
        self.xi = self._get_xi()
        self.gs = self._get_gs()
        self.thetM = self._get_thetM()
        return


class RecursivePrice(NonRecursivePriceGeneric):
    def __init__(
        self,
        mat: ExponentialMat,
        toep: Toeplitz,
        grid: GridParamsGeneric,
        num_params: NumericalParams,
    ):
        super().__init__(mat=mat, toep=toep, grid=grid, num_params=num_params)

    def _get_val(self, k0: int):
        val1 = self.cont[self.grid.nnot - 1, k0 - 1]
        val2 = self.cont[self.grid.nnot - 1, k0]
        return val1, val2

    def _init_variables(self):
        self._thet = self._init_thet()
        self.cont = self._create_cont()
        return

    def _init_thet(self):
        self._init_thet_constants()
        thet = np.zeros((self.grid.K, self.num_params.m0))
        thet[self.grid.nbar - 1, 0] = self._init_thet_sub1()
        thet[self.grid.nbar, 0] = self._init_thet_sub2()
        thet[self.grid.nbar + 1 : self.grid.K, 0] = self._init_thet_sub3()
        return thet

    def _init_thet_sub3(self):
        return (
            np.exp(
                self.grid.xmin
                + self.grid.dx * np.arange(self.grid.nbar + 1, self.grid.K)
            )
            * self.grid.S0
            * self.grid.pf_cons.varthet_star
            - self.grid.W
        )

    def _init_thet_sub2(self):
        return self.grid.W * (
            np.exp(self.grid.dx - self.grid.rho)
            * (self.grid.pf_cons.varthet_01 + self.d1)
            - (0.5 + self.dbar1)
        )

    def _init_thet_sub1(self):
        return self.grid.W * (np.exp(-self.grid.rho) * self.d0 - self.dbar0)

    def _init_thet_constants(self):
        self._init_sigma()
        self._init_es()
        self._init_dbar()
        self._init_d()

    def _init_d(self):
        self.d0 = self._get_d0()
        self.d1 = self._get_d1()

    def _init_dbar(self):
        self.dbar0 = 0.5 + self.grid.zeta * (0.5 * self.grid.zeta - 1)
        self.dbar1 = self.sigma * (1 - 0.5 * self.sigma)

    def _init_es(self):
        self.es1 = np.exp(self.grid.dx * self.sigma_plus)
        self.es2 = np.exp(self.grid.dx * self.sigma_minus)

    def _init_sigma(self):
        self.sigma = 1 - self.grid.zeta
        self.sigma_plus = (self.grid.gauss_quad.q_plus - 0.5) * self.sigma
        self.sigma_minus = (self.grid.gauss_quad.q_minus - 0.5) * self.sigma

    def _get_d1(self):
        return (
            np.exp((self.grid.rho - self.grid.dx) * 0.5)
            * self.sigma
            / 18
            * (
                5
                * (
                    (0.5 * (self.grid.zeta + 1) + self.sigma_minus) * self.es2
                    + (0.5 * (self.grid.zeta + 1) + self.sigma_plus) * self.es1
                )
                + 4 * (self.grid.zeta + 1)
            )
        )

    def _get_d0(self):
        return (
            np.exp((self.grid.rho + self.grid.dx) * 0.5)
            * self.sigma**2
            / 18
            * (
                5
                * (
                    (1 - self.grid.gauss_quad.q_minus) * self.es2
                    + (1 - self.grid.gauss_quad.q_plus) * self.es1
                )
                + 4
            )
        )

    def _update_thet(self, j: int) -> None:
        self._thet[0, j] = self._update_thet_sub1(j)
        self._thet[-1, j] = self._update_thet_sub2(j)
        self._thet[1 : self.grid.K - 1, j] = self._update_thet_sub3(j)
        return

    def _update_thet_sub3(self, j):
        return (
            self.cont[0 : self.grid.K - 2, j]
            + 10 * self.cont[1 : self.grid.K - 1, j]
            + self.cont[2 : self.grid.K, j]
        ) / 12

    def _update_thet_sub2(self, j):
        return (
            13 * self.cont[-1, j]
            + 15 * self.cont[-2, j]
            - 5 * self.cont[-3, j]
            + self.cont[-4, j]
        ) / 48

    def _update_thet_sub1(self, j):
        return (
            13 * self.cont[0, j]
            + 15 * self.cont[1, j]
            - 5 * self.cont[2, j]
            + self.cont[3, j]
        ) / 48

    def _vec_for_thet_update(self):
        return self._thet[: self.grid.K, 0]


class ProjBarrierPricer_SV:
    def __init__(self, model: FourierModel, N: int = 2**11):
        self._init_constants(N)
        self.model = model

    def _init_constants(self, N: int):
        self.N = N

    # Main funtion: pricing method
    def price(
        self, T: float, W: int, S0: float, M: int, H: int, down: bool, is_call: bool
    ) -> float:
        self._check_call_put(is_call=is_call)
        self.initialization(T, W, S0, M, H, down)
        price = self.recursion.recursive_price()
        return price

    def initialization(self, T: float, W: int, S0: float, M: int, H: int, down: bool):
        self.num_params = NumericalParams()
        self.grid = GridParams(
            W, S0, self.N, alpha=self.num_params.alpha, T=T, M=M, H=H, down=down
        )
        self.exp_mat = ExponentialMat(self.grid, self.model, self.num_params, T)
        self.toep = Toeplitz(self.grid, self.model)
        self.recursion = RecursivePrice(
            self.exp_mat, self.toep, self.grid, self.num_params
        )
        return

    def _beta_computation(self):
        return

    def _check_call_put(self, is_call: bool):
        match is_call:
            case False:
                raise NotImplementedError("Only put pricing is implemented so far.")
            case True:
                pass
