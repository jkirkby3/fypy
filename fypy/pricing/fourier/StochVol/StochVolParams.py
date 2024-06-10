import numpy as np
import scipy

from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Union, Any, Optional
from fypy.model.sv.Heston import _HestonBase
from fypy.model.sv.HestonDEJumps import HestonDEJumps
from fypy.model.sv.Bates import Bates
from fypy.model.sv.Heston import Heston

from fypy.model.FourierModel import FourierModel
from copy import deepcopy

# from typing import Union


# From a list of array, return a tensor whose slcies are diagonal with the vector as diag
def diags_to_tens(vector_list: np.ndarray) -> np.ndarray:
    vectors = np.array(vector_list)
    tensor_shape = (*vectors.shape[:-1], vectors.shape[-1], vectors.shape[-1])
    tensor = np.zeros(tensor_shape, dtype="complex_")
    idx = np.arange(vectors.shape[-1])
    tensor[..., idx, idx] = vectors
    return tensor


# Scaling trick to compute exponential matrix without getting NaN
def custom_expm(matrix: np.ndarray, n: int = 4) -> np.ndarray:
    return np.linalg.matrix_power(scipy.linalg.expm(matrix / 2**n), 2**n)


@dataclass
class NumericalParams:
    m0: int = 30
    alpha: float = 6
    gamma: float = 3.3
    gridMethod: int = 4
    gridMultParam: float = 0.2
    boundaryMethod: int = 1


@dataclass
class GaussianQuad:
    q_plus: float = 0.5 * (1 + (3 / 5) ** 0.5)
    q_minus: float = 0.5 * (1 - (3 / 5) ** 0.5)
    b3: float = 15**0.5
    b4: float = b3 / 10


class PayoffConstants:
    def __init__(self, dx: float, b4: float, b3: float):
        self.varthet_01 = (
            np.exp(0.5 * dx) * (5 * np.cosh(b4 * dx) - b3 * np.sinh(b4 * dx) + 4) / 18
        )
        self.varthet_m10 = (
            np.exp(-0.5 * dx) * (5 * np.cosh(b4 * dx) + b3 * np.sinh(b4 * dx) + 4) / 18
        )
        self.varthet_star = self.varthet_01 + self.varthet_m10


# TODO: definir zeta et rho comme des classes à implémenter
class GridParamsGeneric(ABC):
    def __init__(
        self,
        W: float,
        S0: float,
        N: int,
        alpha: float,
        T: float,
        M: int,
        H: Optional[int] = None,
        down: Optional[bool] = None,
    ):
        # required variables
        self.W = W
        self.S0 = S0
        self.lws = np.log(W / S0)
        self.N = N
        self.alpha = alpha
        self.H = H
        self.T = T
        self.M = M
        self.dt = T / M
        self.down = down
        self.gauss_quad = GaussianQuad()
        # to be init later
        self.A = 0
        self.dx = 0
        self.K = 0
        self.a = 0
        self.rho = 0
        self.dxi = 0
        self.xmin = 0
        self.xbar = 0
        self.nbar = 0
        self.xi = np.ndarray([])
        self.upsilon = 0
        self.C_an = 0
        self.klf = 0
        self.klc = 0
        self.lf = 0
        self.nnot = 0
        self.gs = np.array([])
        self.zeta = 0
        self.pf_cons = PayoffConstants(self.dx, self.gauss_quad.b4, self.gauss_quad.b3)
        self.thetM = np.ndarray([])

    @abstractmethod
    def init_variables(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def conditional_variables(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_and_create_variables(self):
        raise NotImplementedError

    def _get_nnot(self):
        return int(self.K / 2)

    def _get_xi(self):
        return self.dxi * np.arange(self.N)

    def _get_zeta(self):
        return self.a * self.rho

    def _get_rho(self):
        return self.lws - (self.xmin + (self.nbar - 1) * self.dx)

    def _get_K(self):
        return int(self.N / 2)

    def _get_a(self):
        return 1 / self.dx

    def _get_dx(self):
        return 2 * self.alpha / (self.N - 1)

    def _get_xmin(self) -> float:
        return (1 - self.K / 2) * self.dx

    def _get_dxi(self) -> float:
        return 2 * np.pi * self.a / self.N

    def _get_nbar(self) -> int:
        return int(self.a * (self.lws - self.xmin) + 1)

    def _get_dxtil(self) -> float:
        return self.dx

    def _get_gs(self) -> np.ndarray:
        try:
            gs = np.zeros(self.K)
            gs[: self.nbar] = self.W - self.S0 * np.exp(
                self.xmin + self.dx * np.arange(self.nbar)
            )
            return gs
        except:
            raise NotImplementedError

    def _get_thetM(self) -> np.ndarray:
        try:
            thetM = np.zeros(self.K)
            thetM[: self.nbar - 1] = self.W - self.pf_cons.varthet_star * (
                self.S0 * np.exp(self.xmin + self.dx * np.arange(self.nbar - 1))
            )
            thetM[self.nbar - 1] = self.W * (0.5 - self.pf_cons.varthet_m10)
            return thetM
        except:
            raise ValueError

    #########################################
    ######## Constants for grid bounds ######
    #########################################

    def _get_var_grid_bound(
        self, model: FourierModel, T: float, gamma: float
    ) -> tuple[float, float]:
        if isinstance(model, _HestonBase):
            mu_h = np.exp(-model.kappa * T) * model.v_0 + model.theta * (
                1 - np.exp(-model.kappa * T)
            )
            sig2_h = self._get_sigma_heston(model, T)
            lx = max(0.00001, mu_h - gamma * sig2_h**0.5)
            ux = mu_h + gamma * sig2_h**0.5
            return ux, lx
        else:
            raise NotImplementedError("Only implemented for Heston so far.")

    def _get_sigma_heston(self, model: _HestonBase, T: float) -> float:
        return (
            model.sigma_v**2
            / (model.kappa)
            * (
                model.v_0 * (np.exp(-model.kappa * T) - np.exp(-2 * model.kappa * T))
                + 0.5
                * model.theta
                * (1 - 2 * np.exp(-model.kappa * T) + np.exp(-2 * model.kappa * T))
            )
        )

    def _get_mu_heston(self, model: _HestonBase, T: float):
        return np.exp(-model.kappa * T) * model.v_0 + model.theta * (
            1 - np.exp(-model.kappa * T)
        )


class ExponentialMat:
    def __init__(
        self,
        grid: GridParamsGeneric,
        model: FourierModel,
        num_params: NumericalParams,
        T: float,
    ):
        self.grid = grid
        self.model = model
        self.boundary_method = 1
        self.num_params = num_params
        self.T = T
        self.ux, self.lx = self.grid._get_var_grid_bound(
            self.model, self.T / 2, self.num_params.gamma
        )

    def get_v(self) -> np.ndarray:
        # common parameters for all gridMethod
        m0 = self.num_params.m0
        nx = m0
        dx = (self.ux - self.lx) / nx
        # TODO: check if model is Heston
        center = self.model.v_0
        # different gridMethod case
        match self.num_params.gridMethod:
            case 1:
                return self.lx + dx * np.arange(1, nx + 1)
            case 4:
                v = self._get_v_grid4(m0, center)
                return v
            case _:
                raise NotImplementedError("Only gridMethod [1,4] implemented so far.")

    def _get_v_grid4(self, m0: int, center: float) -> np.ndarray:
        v = np.zeros(m0)
        v[0] = self.lx
        v[-1] = self.ux
        alpha_mult = self.num_params.gridMultParam * (v[-1] - v[0])
        c1, c2 = np.arcsinh((v[[0, -1]] - center) / alpha_mult)
        sub_vec = c2 / m0 * np.arange(2, m0) + c1 * (1 - np.arange(2, m0) / m0)
        v[1:-1] = center + alpha_mult * np.sinh(sub_vec)
        return v

    ###################################
    ####### Build the matrix Q ########
    ###################################

    def _get_Q(self, v: np.ndarray) -> np.ndarray:
        m0 = self.num_params.m0
        mu_vec = self._mu_func(v)
        mu_plus, mu_minus = mu_vec * (mu_vec > 0), -mu_vec * (-mu_vec > 0)
        sig2 = self._sig_func(v) ** 2
        h = np.diff(v)

        match self.num_params.gridMethod:
            case 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9:
                Q = self._get_Q_non_unif(v, mu_plus, mu_minus, sig2, h)
            case _:
                raise NotImplementedError("Only gridMethod [1-> 9] implemented so far.")

        match self.num_params.boundaryMethod:
            case 1:
                return self._get_Q_boundaries(m0, Q, mu_vec, h)
            case _:
                raise NotImplementedError("Only boundMethod 1 implemented so far.")

    def _get_Q_non_unif(
        self,
        v: np.ndarray,
        mu_plus: np.ndarray,
        mu_minus: np.ndarray,
        sig2: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        zeros = np.zeros(len(v) - 2)
        sub_vec = sig2[1:-1] - (h[1:] * mu_plus[1:-1] + h[:-1] * mu_minus[1:-1])
        aa = np.max([sub_vec, zeros], axis=0) / (h[:-1] + h[1:])
        sub_diag = (mu_minus[1:-1] + aa) / h[:-1]
        up_diag = (mu_plus[1:-1] + aa) / h[1:]
        diag = np.hstack((0, sub_diag + up_diag, 0))
        sub_diag = np.hstack((sub_diag, 0))
        up_diag = np.hstack((0, up_diag))
        Q = np.diag(sub_diag, k=-1) + np.diag(up_diag, k=+1) - np.diag(diag)
        return Q

    def _get_Q_boundaries(
        self, m0: int, Q: np.ndarray, mu_vec: np.ndarray, h: np.ndarray
    ) -> np.ndarray:
        Q[0, 1] = np.abs(mu_vec[0]) / h[0]
        Q[0, 0] = -Q[0, 1]
        Q[m0 - 1, m0 - 2] = np.abs(mu_vec[-1]) / h[-1]
        Q[m0 - 1, m0 - 1] = -Q[m0 - 1, m0 - 2]
        return Q

    def _mu_func(self, u: np.ndarray) -> np.ndarray:  ##
        if isinstance(self.model, _HestonBase):
            return self.model.kappa * (self.model.theta - u)
        else:
            raise NotImplementedError("Only implemented for Heston so far.")

    def _sig_func(self, u: np.ndarray) -> np.ndarray:
        if isinstance(self.model, _HestonBase):
            return self.model.sigma_v * u**0.5

        else:
            raise NotImplementedError("Only implemented for Heston so far.")

    ###################################
    #######    END Matrix Q    ########
    ###################################

    def get_exp_tensor(self) -> np.ndarray:
        v = self.get_v()
        fv = self._get_fv(v)
        exp_A = self._get_exp_diags(v)
        lambda_exp_power = self._get_lambda_exp(fv)
        exp_A[..., 1:] = exp_A[..., 1:] * lambda_exp_power
        return exp_A

    def _get_exp_diags(self, v: np.ndarray):
        Q = self._get_Q(v)
        v1 = self._get_v1(v)
        v2 = self._get_v2(v)
        Qt = self.grid.dt * Q.T
        diags_mat = (
            self.grid.xi[:, None] @ v1[None, :]
            - (self.grid.xi.T[:, None] ** 2) @ v2[None, :]
            + self.grid.dt * self.model.psi_J(self.grid.xi)[:, None]
        )
        diag_tensor = diags_to_tens(diags_mat) + Qt
        exp_diags = custom_expm(diag_tensor)
        return exp_diags.transpose(1, 2, 0)

    def _get_lambda_exp(self, fv: np.ndarray) -> np.ndarray:
        m0 = self.num_params.m0
        hor_fv = np.repeat(fv[:, None], m0, axis=1)
        vert_fv = np.repeat(fv[None, :], m0, axis=0)
        upper_part = np.triu(hor_fv - vert_fv, k=1)
        lower_part = -upper_part.T
        lambda_dxi = lower_part + upper_part
        lambda_exp = np.exp(lambda_dxi)
        lambda_exp_tens = np.repeat(lambda_exp[None, :], self.grid.N - 1, axis=0)
        lambda_exp_power = np.power(
            lambda_exp_tens, np.arange(1, self.grid.N)[..., None, None]
        )
        return lambda_exp_power.transpose(1, 2, 0)

    def _get_v1(self, v: np.ndarray) -> np.ndarray:
        if isinstance(self.model, _HestonBase):
            c1 = (self.model.rho * self.model.kappa / self.model.sigma_v) - 0.5
            rn_drift = self.model.forwardCurve.drift(0, self.grid.T)
            c2 = (
                rn_drift
                - self.model.rho
                * self.model.theta
                * self.model.kappa
                / self.model.sigma_v
            )
            return self.grid.dt * 1j * (c1 * v + c2 - self.model.psi_J(-1j))
        else:
            raise NotImplementedError("Only implemented for Heston so far.")

    def _get_v2(self, v: np.ndarray) -> np.ndarray:
        if isinstance(self.model, _HestonBase):
            c3 = 0.5 * (1 - self.model.rho**2)
            return self.grid.dt * c3 * v
        else:
            raise NotImplementedError("Only implemented for Heston so far.")

    def _get_fv(self, v: np.ndarray) -> np.ndarray:
        if isinstance(self.model, _HestonBase):
            fv = (1j * self.grid.dxi * self.model.rho / self.model.sigma_v) * v
            return fv
        else:
            raise NotImplementedError("Only implemented for Heston so far.")


class Toeplitz:
    def __init__(self, grid: GridParamsGeneric, model: FourierModel):
        self.model = model
        self.grid = grid

    def get_beta(self, exp_mat: np.ndarray):
        grand_tens = exp_mat[:, :, 1:] * self.hvec
        stacked = np.dstack((exp_mat[:, :, [0]] / (24 * self.grid.a**2), grand_tens))
        beta_tens = self.cons * np.real(np.fft.fft(stacked, axis=-1))
        first_range = beta_tens[..., self.grid.K - 1 :: -1]
        second_range = beta_tens[..., 2 * self.grid.K - 2 : self.grid.K - 1 : -1]
        toepM = np.dstack(
            (
                first_range,
                np.zeros((first_range.shape[0], first_range.shape[0])),
                second_range,
            )
        )
        beta_fft = np.fft.fft(toepM)
        return beta_fft

    @property
    def zmin(self):
        return (1 - self.grid.K) * self.grid.dx

    @property
    def hvec(self):
        return (
            np.exp(-1j * self.zmin * self.xi)
            * (np.sin(self.xi / (2 * self.grid.a)) / self.xi) ** 2
            / (2 + np.cos(self.xi / self.grid.a))
        )

    @property
    def xi(self):
        return self.grid.dxi * np.arange(1, self.grid.N)

    @property
    def cons(self):
        r = self.model.forwardCurve.drift(0, self.grid.T)
        return 24 * self.grid.a**2 * np.exp(-r * self.grid.dt) / self.grid.N


class AlphaRecursiveReturn:
    def __init__(self, model: FourierModel, T: float, L: float):
        self._model = model
        self.T = T
        self._L = L

    def _get_alpha_variance(self):
        if isinstance(self._model, _HestonBase):
            t = self.T / 2
            eta = self._model.kappa
            v0 = self._model.v_0
            theta = self._model.theta
            mu_h = (np.exp(-eta * t) * v0 + theta * (1 - np.exp(-eta * t))) ** 0.5
            c2 = self._model.c2_jump + (mu_h) ** 2
            alpha = max(
                self._L * (c2 * t + (self._model.c4_jump * self.T) ** 0.5) ** 0.5, 0.5
            )
            return alpha
        else:
            raise NotImplementedError

    def __call__(self):
        return self._get_alpha_variance()


class AddJumpsCharacteristics:
    def __init__(self, model: FourierModel):
        self.model = model
        self._add_psi_fun_cumul()

    def _add_psi_fun_cumul(self):
        if isinstance(self.model, TYPES.HesDE):
            lam, p_up, eta1, eta2 = (
                self.model.lam,
                self.model.p_up,
                self.model.eta1,
                self.model.eta2,
            )
            fun = lambda xi: lam * (
                (1 - p_up) * eta2 / (eta2 + 1j * xi)
                + p_up * eta1 / (eta1 - 1j * xi)
                - 1
            )
            self.model.c2_jump = 2 * lam * p_up / (eta1**2) + 2 * lam * (1 - p_up) / (
                eta2**2
            )
            self.model.c4_jump = 24 * lam * (p_up / eta1**4 + (1 - p_up) / eta2**4)
            self.model.psi_J = fun

        elif isinstance(self.model, TYPES.HesDE):
            lam, muj, sigj = self.model.lam, self.model.muj, self.model.sigj
            fun = lambda xi: lam * (np.exp(1j * xi * muj - 0.5 * sigj**2 * xi**2) - 1)
            self.model.c2_jump = lam * (muj**2 + sigj**2)
            self.model.c4_jump = lam * (
                muj**4 + 6 * sigj**2 * muj**2 + 3 * sigj**4 * lam
            )
            self.model.psi_J = fun

        elif isinstance(self.model, TYPES.Hes):
            fun = lambda xi: 0 * (np.array(xi) > 0)
            self.model.psi_J = fun
            self.model.c2_jump = 0
            self.model.c4_jump = 0

        else:
            raise NotImplementedError("Model")
        return

    def get_model(self):
        return self.model


@dataclass
class TYPES:
    HesDE = HestonDEJumps
    Bates = Bates
    Hes = Heston
    Hes_base = _HestonBase
