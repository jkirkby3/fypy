import numpy as np

# np.set_printoptions(precision=4)

from fypy.model.FourierModel import FourierModel

from copy import deepcopy


from fypy.pricing.fourier.StochVol.StochVolParams import (
    GridParamsGeneric,
    NumericalParams,
    ExponentialMat,
    AlphaRecursiveReturn,
    TYPES,
    AddJumpsCharacteristics,
)
from fypy.pricing.fourier.StochVol.StochVolPricer import RecursiveReturnPricer


class GridParams(GridParamsGeneric):
    def __init__(self, W: float, S0: float, N: int, alpha: float, T: float, M: int):
        super().__init__(W, S0, N, alpha, T, M)
        self.init_variables()

    def init_variables(self, **kwargs):
        self.dx = 2 * self.alpha / (self.N - 1)
        self.a = 1 / self.dx
        self.dt = self.T / self.M
        self.xmin = (1 - self.N / 2) * self.dx
        self.A = 32 * self.a**4
        self.C_an = self.A / self.N
        self.dxi = 2 * np.pi * self.a / self.N
        self.xi = self.dxi * np.arange(self.N)
        self.hlocalCF = lambda x: x**2
        return

    def conditional_variables(self):
        return

    def update_and_create_variables(self):
        return


class RecursivePricer(RecursiveReturnPricer):
    def __init__(
        self,
        model: FourierModel,
        mat: ExponentialMat,
        grid: GridParamsGeneric,
        num_params: NumericalParams,
    ):
        super().__init__(model, mat, grid, num_params)

    def _set_left_and_NMM(self, contract: int):
        match contract:
            case 1 | 3:
                self.leftGridPoint = self.grid.xmin
                self.NNM = self.grid.N
            case _:
                raise NotImplementedError(
                    "Only contracts 1 and 3 are implemented for now."
                )
        return

    def _init_phi(self):
        self.phi_old = np.zeros((self.grid.N - 1, self.num_params.m0), dtype="complex_")
        self.phi_y_new = np.zeros(
            (self.grid.N - 1, self.num_params.m0), dtype="complex_"
        )
        return

    def _get_phy_new(self):
        self.phi_old = self.transit_mat[:, :, 1:].sum(axis=0).T
        to_fft = np.vstack(
            (
                np.asarray([1 / self.grid.A] * self.num_params.m0),
                self.phi_old * self.hvec[:, None],
            )
        ).T
        beta_temp = np.real(np.fft.fft(to_fft))
        self.phi_y_new = self.psi @ beta_temp.T
        return

    def _get_phi(self):
        to_fft = deepcopy(self.transit_mat)
        to_fft[:, :, 0] = to_fft[:, :, 0] / self.grid.A
        to_fft[:, :, 1:] = to_fft[:, :, 1:] * self.hvec[None, None, :]
        beta_temp = np.real(np.fft.fft(to_fft))
        self.phi_y = np.matmul(beta_temp.transpose(1, 0, 2), self.psi.T)
        return

    def _get_PhiY(self):
        self._PSI_computation(1, self.grid.a)
        self.transit_mat = self.exp_mat.get_exp_tensor()
        self.zeta = self._get_zeta_vec(self.grid.a, self.grid.xi)
        self.grid.xi = self.grid.xi[1:]
        self.hvec = np.exp(-1j * self.grid.xmin * self.grid.xi) * self.zeta
        self._init_phi()
        self._get_phy_new()
        self._get_phi()
        return


class ProjDVSwap_SV:
    def __init__(self, model: FourierModel, N: int = 2**10, L: float = 10.0):
        self.L = L
        self.N = N
        self.model = AddJumpsCharacteristics(model).get_model()

    def _beta_computation(self, k: int):
        beta = np.real(
            np.fft.fft(
                np.hstack(
                    (1 / self.grid.A, self.hvec * self.recursive_pricer.phi_y_new[:, k])
                )
            )
        )
        return beta

    def price(
        self,
        T: float,
        M: int,
        W: float,
        contract: int,
    ):
        self._init_constants(W, T, M)
        self.recursive_pricer._recursion()
        price = self._interpolation(contract)
        return price

    def _init_constants(self, K: float, T: float, M: float):
        self._init_classes(K, T, M)
        self.recursive_pricer._get_PhiY()
        return

    def _init_classes(self, K, T, M):
        alpha = AlphaRecursiveReturn(self.model, T, self.L)()
        self.num_params = NumericalParams()
        self.grid = GridParams(W=K, N=self.N, alpha=alpha, T=T, M=M, S0=1)
        self.exp_mat = ExponentialMat(self.grid, self.model, self.num_params, T)
        self.recursive_pricer = RecursivePricer(
            self.model, self.grid, self.num_params, self.exp_mat
        )

    def _adjust_xmin_interp(self, contract: int):
        match contract:
            case 1:
                self.grid.xmin = -self.grid.dx
            case 3:
                self.grid.xmin = self.grid.W * self.grid.T - self.grid.dx
            case _:
                raise NotImplementedError("No other contracts for now.")
        return

    def _set_hvec(self):
        match bool(self.grid.dx != 0):
            case True:
                self.hvec = (
                    np.exp(-1j * self.grid.xmin * self.grid.xi)
                    * self.recursive_pricer.zeta
                )
            case False:
                self.hvec = self.recursive_pricer.zeta
        return

    def _get_val(self, k: int, grid: np.ndarray):
        beta_temp = self._beta_computation(k)
        val = (
            self.grid.C_an
            * grid[: int(self.grid.N / 2)]
            @ beta_temp[: int(self.grid.N / 2)]
        )
        return val

    def _interpolation(self, contract: int):
        if isinstance(self.model, TYPES.Hes_base):
            k0 = self.recursive_pricer._get_k0()
            v0 = self.model.v_0
            v = self.exp_mat.get_v()
            disc = self.model.discountCurve.discount_T(self.grid.T)
            val1, val2 = self._get_vals(contract, k0)
            price = val1 + (val2 - val1) * (v0 - v[k0 - 1]) / (v[k0] - v[k0 - 1])
            match contract:
                case 3:
                    return disc * price / self.grid.T
                case 1:
                    return price / self.grid.T
        else:
            raise NotImplementedError

    def _get_vals(self, contract, k0):
        self._adjust_xmin_interp(contract)
        self._set_hvec()
        grid = self._get_grid_interp()
        val1 = self._get_val(k0 - 1, grid)
        val2 = self._get_val(k0, grid)
        return val1, val2

    def _get_grid_interp(self):
        grid = self.grid.dx * np.arange(-1, self.grid.N - 1)
        grid[0] = grid[0] / 24 + self.grid.dx / 20
        grid[1] = 7 * self.grid.dx / 30
        grid[2] = grid[2] * 23 / 24 + self.grid.dx / 20
        return grid
