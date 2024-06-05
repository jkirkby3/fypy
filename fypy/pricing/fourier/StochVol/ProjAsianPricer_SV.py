import numpy as np

# np.set_printoptions(precision=4)

from fypy.model.FourierModel import FourierModel


from scipy.io import loadmat as loadmat
from fypy.pricing.fourier.StochVol.StochVolParams import (
    GridParamsGeneric,
    NumericalParams,
    ExponentialMat,
    AddJumpsCharacteristics,
)

from fypy.pricing.fourier.StochVol.StochVolPricer import RecursiveReturnPricer

# TODO: check sum axis, refactor, details arguments x=x etc...
# TODO: refactorer les self.a (doublon entre params et pricer)
# TODO: upsilon gamma ? voir le nom et où il est défini
# TODO: lier les coefs v aux coefs de ProjPricer et d'autres méthodes (zeta etc...)


class GridParams(GridParamsGeneric):
    def __init__(
        self,
        W: float,
        S0: float,
        N: int,
        alpha: float,
        T: float,
        M: int,
        P: int,
        Pbar: int,
    ):
        super().__init__(W, S0, N, alpha, T, M)
        self.init_variables(P, Pbar)

    def init_variables(self, P: int, Pbar: int):
        self.P = P
        self.Pbar = Pbar
        self.a = 2**P
        self.C_an = 1
        self.dx = 1 / self.a
        self.dxi = self._get_dxi()
        self.xi = self._get_xi()
        self.hlocalCF = lambda x: np.log(1 + np.exp(x))
        self.upsilon = (32 * self.a**4) / self.N
        return

    def conditional_variables(self):
        return

    def update_and_create_variables(self):
        return


class RecursivePrice(RecursiveReturnPricer):
    def __init__(
        self,
        model: FourierModel,
        mat: ExponentialMat,
        grid: GridParamsGeneric,
        num_params: NumericalParams,
    ):
        super().__init__(model=model, exp_mat=mat, grid=grid, num_params=num_params)
        self.transit_mat = self.exp_mat.get_exp_tensor()

    def _set_left_and_NMM(self):
        self._get_mum_xm_nm_vec(self.grid)
        self.leftGridPoint = self.xm[0]
        self.NNM = int(self.grid.N + self.nm_vec[self.grid.M - 2])

    def _get_PhiY(self):
        self.zeta = self._get_zeta_vec(self.grid.a, self.grid.xi)
        self.grid.xi = self.grid.xi[1:]
        self.chf = np.sum(self.transit_mat, axis=0).T
        self._PSI_computation(None, self.grid.a)
        self.psi = np.vstack((np.ones(self.psi.shape[1]), self.psi))
        self.psi[-1, :] = 0

    def _beta_computation(self, xm: int, j: int):
        gamma = 1 / (32 * self.grid.a**4)
        h = np.zeros(self.grid.N, dtype="complex_")
        h[0] = gamma
        h[1:] = self.chf[1:, j] * self.zeta * np.exp(-1j * xm * self.grid.xi)
        return np.real(np.fft.fft(h))

    def _recursion(self):
        M = self.grid.M
        for m in np.arange(2, M + 1):
            idx_m = m - 2
            self.phi_z = self._get_phi_z(idx_m)
            self._update_chf()
        return

    def _get_phi_z(self, idx_m: int):
        phi_z = []
        for j in range(self.num_params.m0):
            beta = self._beta_computation(self.xm[idx_m], j)
            phi_z_j = self.grid.upsilon * (
                self.psi[
                    :,
                    int(self.nm_vec[idx_m]) : (int(self.nm_vec[idx_m]) + self.grid.N),
                ]
                @ beta
            )
            phi_z.append(phi_z_j)
        return np.array(phi_z).T

    def _update_chf(self):
        for j in range(self.num_params.m0):
            self.chf[:, j] = self.phi_z[:, 0] * self.transit_mat[0, j, :]
            for k in range(1, self.num_params.m0):
                self.chf[:, j] += self.phi_z[:, k] * self.transit_mat[k, j, :]

    # µm, nm, xm vectors (equations 42 and 43)
    def _get_mum_xm_nm_vec(self, grid: GridParams):
        dx = self.grid.dx
        theta = grid.dt * self._model.forwardCurve.drift(0, grid.T)
        self._get_nm_vec(theta=theta)
        self.nm_vec = np.hstack((0, self.nm_vec))
        self.mu_vec = theta + self.nm_vec * dx
        self.xm = self.mu_vec + (1 - 0.5 * self.grid.N) * dx

    def _get_nm_vec(self, theta: np.ndarray):
        m_vec = np.arange(2, self.grid.M + 1)
        # if theta = 0, do tayor development
        match bool(np.abs(theta) >= 1e-8):
            case True:
                self.nm_vec = np.floor(
                    self.grid.a
                    * np.log((np.exp(m_vec * theta) - 1) / (np.exp(theta) - 1))
                ).astype(int)
            case False:
                self.nm_vec = np.floor(self.grid.a * np.log(m_vec)).astype(int)


class ProjAsianPricer_SV:
    def __init__(self, model: FourierModel, P: int = 2**11, Pbar: int = 2**5):
        self._init_constants(P, Pbar)
        # because jump chf is not implemented
        self.model = AddJumpsCharacteristics(model).get_model()

    ####################################################
    ######## 3 steps: Init, Recursion, Valuation #######
    ####################################################

    # Main funtion: pricing method
    def price(self, T: float, W: int, S0: float, M: float, is_call: bool) -> float:
        self._initialization(T, W, S0, M, is_call)
        self._recursion()
        price = self._valuation_stage()
        return price

    def _initialization(self, T: float, W: int, S0: float, M: float, is_call: bool):
        self._init_classes(T, W, S0, M, is_call)
        self._init_tensors()
        return

    def _init_tensors(self):
        self._get_v_coef()
        self.j0 = self.rec_pricer._get_k0()
        self.zeta = self.rec_pricer._get_zeta_vec(self.grid.a, self.grid.xi)
        self.transit_mat = self.exp_mat.get_exp_tensor()
        self.chf = self._init_chf()

    def _init_classes(self, T: float, W: int, S0: float, M: float, is_call: bool):
        self._check_call_put(is_call=is_call)
        self.num_params = NumericalParams()
        self.grid = GridParams(
            W, S0, self.N, self.num_params.alpha, T, M, self.P, self.Pbar
        )
        self.exp_mat = ExponentialMat(self.grid, self.model, self.num_params, T)
        self.rec_pricer = RecursivePrice(
            self.model, self.exp_mat, self.grid, self.num_params
        )
        return

    def _recursion(self):
        self.rec_pricer._get_PhiY()
        self.rec_pricer._recursion()

    def _valuation_stage(self):
        self._set_phi_y_phi_z()
        self._set_valuation_constants()
        pc_price = self._valuation_sub()
        return pc_price

    def _valuation_sub(self):
        val1, val2 = self._get_vals()
        val = self._interp(val1, val2)
        pc_price = self._pc_price(val)
        return pc_price

    def _set_phi_y_phi_z(self):
        self.chf = self.rec_pricer.chf
        self.phi_z = self.rec_pricer.phi_z

    def _set_valuation_constants(self):
        y_star = np.log(((self.grid.M + 1) * self.grid.W / self.grid.S0) - 1)
        self.k_star = self._get_k_star(self.grid.dx, y_star)
        self.gk = self._get_g_coef(
            self.k_star, y_star, self.grid.S0, self.grid.W, self.grid.M
        )
        self.xM = y_star - (self.k_star - 1) * self.grid.dx

    def _get_k_star(self, dx, y_star):
        return int(
            np.floor(
                1
                + self.grid.a
                * (y_star - (self.rec_pricer.mu_vec[-1] + (1 - self.grid.N / 2) * dx))
            )
        )

    def _pc_price(self, val: float):
        C = self.grid.S0 / (self.grid.M + 1)
        r = -np.log(self.model.discountCurve.discount_T(self.grid.T)) / self.grid.T
        q = -np.log(self.model.forwardCurve._divDiscount(self.grid.T)) / self.grid.T
        T = self.grid.T
        M = self.grid.M
        pc_price = (
            val
            + C
            * np.exp(-r * T)
            * (np.exp((r - q) * T * (1 + 1 / M)) - 1)
            / (np.exp((r - q) * self.grid.dt) - 1)
            - self.grid.W * np.exp(-r * T)
        )
        return pc_price

    def _interp(self, val1: float, val2: float):
        val = val1 + (val2 - val1) * (
            self.model.v_0 - self.exp_mat.get_v()[self.j0 - 1]
        ) / (self.exp_mat.get_v()[self.j0] - self.exp_mat.get_v()[self.j0 - 1])
        return val

    def _get_vals(self):
        disc = self.model.discountCurve.discount_T(self.grid.T)
        val1 = self._get_val(self.j0 - 1)
        val2 = self._get_val(self.j0)
        return disc * val1, disc * val2

    def _get_val(self, k: int):
        beta = self.rec_pricer._beta_computation(self.xM, k)
        val2 = self.grid.upsilon * (beta[: self.k_star + 1] * self.gk).sum()
        return val2

    ####################################################
    ####### INITIALIZATION auxiliary functions  ########
    ####################################################

    # pricing parameters input
    def _init_constants(self, P: int, Pbar: int):
        self.P = P
        self.Pbar = Pbar
        self.a = 2**P  # not the same as in the grid
        self.abar = 2**Pbar
        self.N = 2 ** (P + Pbar)

    # get v and vbar coefficients as defined in Table 15 appendix A
    def _get_v_coef(self):
        dx = self.grid.dx
        self.v1 = (1 / 20) * (
            np.exp(-7 / 4 * dx) / 54
            + np.exp(-1.5 * dx) / 18
            + np.exp(-1.25 * dx) / 2
            + 7 * np.exp(-dx) / 27
        )

        self.v0 = (1 / 20) * (
            28 / 27
            + np.exp(-7 / 4 * dx) / 54
            + np.exp(-1.5 * dx) / 18
            + np.exp(-1.25 * dx) / 2
            + 14 * np.exp(-dx) / 27
            + 121 / 54 * np.exp(-0.75 * dx)
            + 23 / 18 * np.exp(-0.5 * dx)
            + 235 / 54 * np.exp(-0.25 * dx)
        )

        self.vm1 = (1 / 90) * (
            (28 + 7 * np.exp(-dx)) / 3
            + (
                14 * np.exp(dx)
                + np.exp(-7 / 4 * dx)
                + 242 * np.cosh(0.75 * dx)
                + 470 * np.cosh(0.25 * dx)
            )
            / 12
            + 0.25
            * (np.exp(-1.5 * dx) + 9 * np.exp(-1.25 * dx) + 46 * np.cosh(0.5 * dx))
        )

        self.vstar = (1 / 90) * (
            14 / 3 * (2 + np.cosh(dx))
            + 0.5
            * (np.cosh(1.5 * dx) + 9 * np.cosh(1.25 * dx) + 23 * np.cosh(0.5 * dx))
            + 1
            / 6
            * (
                np.cosh(7 / 4 * dx)
                + 121 * np.cosh(0.75 * dx)
                + 235 * np.cosh(0.25 * dx)
            )
        )
        self.vbar_star = 1
        self.vbar_m1 = 23 / 24
        self.vbar_0 = 1 / 2
        self.vbar_1 = 1 / 24
        return

    # characteristic function phi_Y initialized
    def _init_chf(self):
        return np.sum(self.transit_mat, axis=0).T

    def _beta_computation(self):
        raise NotImplementedError

    def _get_g_coef(self, k_star: int, y_star: float, S0: float, K: float, M: int):
        self.v0
        C = S0 / (M + 1)
        Ek = np.exp(y_star + (np.arange(1, k_star + 2) - k_star) * self.grid.dx)
        D = K - C
        # equation (47)
        gk = np.zeros(k_star + 1)
        gk[: k_star - 2] = self.vbar_star * D - C * self.vstar * Ek[: k_star - 2]
        gk[k_star - 2] = self.vbar_m1 * D - C * self.vm1 * Ek[k_star - 2]
        gk[k_star - 1] = self.vbar_0 * D - C * self.v0 * Ek[k_star - 1]
        gk[-1] = self.vbar_1 * D - C * self.v1 * Ek[-1]
        return gk

    # check that contract asked is implemented (NOTE: to move in a proper class ?)
    def _check_call_put(self, is_call: bool):
        match is_call:
            case False:
                raise NotImplementedError("Only put pricing is implemented so far.")
            case True:
                pass
        return
