import numpy as np
from abc import abstractmethod
from fypy.model.FourierModel import FourierModel
from typing import Optional, Any

from fypy.pricing.fourier.StochVol.StochVolParams import (
    ExponentialMat,
    Toeplitz,
    GridParamsGeneric,
    NumericalParams,
    TYPES,
)


class RecursiveReturnPricer:
    def __init__(
        self,
        model: FourierModel,
        grid: GridParamsGeneric,
        num_params: NumericalParams,
        exp_mat: ExponentialMat,
    ):
        self._model = model
        self.exp_mat = exp_mat
        self.grid = grid
        self.num_params = num_params

    def _get_k0(self) -> int:
        k0 = 2
        v = self.exp_mat.get_v()
        while self.exp_mat.model.v_0 > v[k0 - 1] and k0 < self.num_params.m0:
            k0 += 1
        k0 -= 1
        return k0

    def _get_zeta_vec(self, a: float, xi: np.ndarray):
        b0 = 1208 / 2520
        b1 = 1191 / 2520
        b2 = 120 / 2520
        b3 = 1 / 2520
        w = xi[1:]
        zeta = np.empty_like(w, dtype=complex)
        zeta = (np.sin(w / (2 * a)) / w) ** 4 / (
            b0 + b1 * np.cos(w / a) + b2 * np.cos(2 * w / a) + b3 * np.cos(3 * w / a)
        )
        return zeta

    def _sig_computation(self):
        # Gaussian quadrature coefs
        g2 = np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 6
        g3 = np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 6
        v1 = 0.5 * 128 / 225
        v2 = 0.5 * (322 + 13 * np.sqrt(70)) / 900
        v3 = 0.5 * (322 - 13 * np.sqrt(70)) / 900
        # gamma coefs: Appendix A.2.1
        sig = np.array(
            [
                -1.5 - g3,
                -1.5 - g2,
                -1.5,
                -1.5 + g2,
                -1.5 + g3,
                -0.5 - g3,
                -0.5 - g2,
                -0.5,
                -0.5 + g2,
                -0.5 + g3,
            ]
        )
        # cubic spline: sig = phi[3](gamma)*w
        sig[0:5] = (sig[0:5] + 2) ** 3 / 6
        sig[5:10] = 2 / 3 - 0.5 * sig[5:10] ** 3 - sig[5:10] ** 2
        # multiplication by w (denoted as v hat)
        sig[np.array([0, 4, 5, 9])] *= v3
        sig[np.array([1, 3, 6, 8])] *= v2
        sig[np.array([2, 7])] *= v1
        return self.grid.C_an * sig

    def _thet_computation(self, dx: float, x1: np.ndarray, Neta: int, Neta5: int):
        # Computation of thet is required for the _PSI_computation method,
        # which itself is essential for the functioning of the _beta_computation method
        g2 = np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 6
        g3 = np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 6
        thet = np.zeros(Neta)
        thet[5 * np.arange(1, Neta5 + 1) - 3] = x1[0] - 1.5 * dx + dx * np.arange(Neta5)
        thet[5 * np.arange(1, Neta5 + 1) - 5] = (
            x1[0] - 1.5 * dx + dx * np.arange(Neta5) - dx * g3
        )
        thet[5 * np.arange(1, Neta5 + 1) - 4] = (
            x1[0] - 1.5 * dx + dx * np.arange(Neta5) - dx * g2
        )
        thet[5 * np.arange(1, Neta5 + 1) - 2] = (
            x1[0] - 1.5 * dx + dx * np.arange(Neta5) + dx * g2
        )
        thet[5 * np.arange(1, Neta5 + 1) - 1] = (
            x1[0] - 1.5 * dx + dx * np.arange(Neta5) + dx * g3
        )
        return thet

    def _set_zz(self, thet: np.ndarray, a: float):
        dxi = 2 * np.pi * a / self.grid.N
        if self.grid.hlocalCF is not None:
            self.zz = np.exp(1j * dxi * self.grid.hlocalCF(thet))
        else:
            raise ValueError("The method hlocalCF should be initialized.")
        return

    def _set_thet(self):
        self.thet = self.zz
        return

    def _update_thet(self):
        self.thet = self.thet * self.zz
        return

    def _build_psi_col(self, sig: np.ndarray, Neta: int, j: int):
        thet = self.thet
        psi_j = []

        for i in range(10):
            psi_ji = list(
                sig[i]
                * (thet[i : Neta - (19 - i) : 5] + thet[(19 - i) : (Neta - i) : 5])
            )
            psi_j.append(psi_ji)

        psi_j = np.array(psi_j).sum(axis=0)
        self.psi[j, :] = psi_j

    def _PSI_computation(
        self, contract: Optional[float] = None, a: Optional[float] = None
    ):
        if contract is None:
            self._set_left_and_NMM()
        else:
            self._set_left_and_NMM(contract=contract)

        if a is not None:
            dx = 1 / a
        else:
            raise ValueError("a should be a real number.")
        NNM = self.NNM  # Number of columns of PSI
        Neta = 5 * NNM + 15  # Sample size
        Neta5 = NNM + 3

        self.thet = self._thet_computation(
            dx=dx, x1=[self.leftGridPoint], Neta=Neta, Neta5=Neta5
        )
        sig = self._sig_computation()
        self._set_zz(self.thet, a)
        self._set_thet()

        # PSI Matrix: 5-Point GAUSSIAN
        self.psi = np.zeros(
            (self.grid.N - 1, NNM), dtype=np.float64
        )  # The first row will remain ones
        self.psi = self.psi.astype(np.complex128)

        for j in range(self.grid.N - 1):
            self._build_psi_col(sig, Neta, j)
            self._update_thet()
        return

    def _recursion(self):
        self.phi_y = self.phi_y.transpose(1, 0, 2)
        for _ in range(2, self.grid.M + 1):
            self.phi_y_new = np.einsum("ab,bca->ac", self.phi_y_new, self.phi_y)
        return

    @abstractmethod
    def _set_left_and_NMM(self, *args, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def get_PhiY(self):
        raise NotImplementedError


class NonRecursivePriceGeneric:
    def __init__(
        self,
        mat: ExponentialMat,
        toep: Toeplitz,
        grid: GridParamsGeneric,
        num_params: NumericalParams,
    ):
        self.mat = mat
        if not isinstance(self.mat.model, TYPES.Hes_base):
            raise NotImplementedError
        self.toep = toep
        self.grid = grid
        self.num_params = num_params
        self.beta = self._get_beta()
        self._init_variables()

    def _get_beta(self) -> np.ndarray:
        exp_mat = self.mat.get_exp_tensor()
        beta = self.toep.get_beta(exp_mat).transpose((1, 0, 2))
        return beta

    def recursive_price(self) -> float:
        self._thet = self._init_thet()
        self.cont = self._create_cont()
        for _ in np.arange(0, self.grid.M - 1)[::-1]:
            self._time_recursion()
        price = self._interp_price()
        return price

    @abstractmethod
    def _init_thet(self):
        raise NotImplementedError

    def _time_recursion(self) -> None:
        for j in range(self.num_params.m0):
            self._update_thet(j)
        self._update_cont()
        return

    @abstractmethod
    def _update_thet(self, j: int) -> None:
        raise NotImplementedError

    ########################
    ###### CONT Array ######
    ########################

    def _update_cont(self) -> None:
        self.cont = np.zeros((self.grid.K, self.num_params.m0))
        for k in range(self.num_params.m0):
            stacked = np.hstack((self._thet[: self.grid.K, k], np.zeros(self.grid.K)))
            thet_temp = np.fft.fft(stacked)
            for j in range(self.num_params.m0):
                p = np.real(np.fft.ifft(self.beta[j, k, :] * thet_temp))
                self.cont[:, j] += p[: self.grid.K]

    # different from ameerican
    def _create_cont(self) -> np.ndarray:
        cont = np.zeros((self.grid.K, self.num_params.m0))
        vec_thet_update = self._vec_for_thet_update()
        stacked = np.hstack((vec_thet_update, np.zeros(self.grid.K)))
        thet_temp = np.fft.fft(stacked)
        for j in range(self.num_params.m0):
            for k in range(self.num_params.m0):
                p = np.real(np.fft.ifft(self.beta[j, k, :] * thet_temp))
                cont[:, j] += p[: self.grid.K]
        return cont

    @abstractmethod
    def _vec_for_thet_update(self):
        raise NotImplementedError

    #########################
    ##### Interp Price ######
    #########################
    def _get_k0(self) -> int:
        k0 = 2
        v = self.mat.get_v()
        if isinstance(self.mat.model, TYPES.Hes_base):
            while self.mat.model.v_0 > v[k0 - 1] and k0 < self.num_params.m0:
                k0 += 1
            k0 -= 1
            return k0

        else:
            raise NotImplementedError

    def _interp_price(self) -> float:
        k0 = self._get_k0()
        v = self.mat.get_v()
        val1, val2 = self._get_val(k0)
        if isinstance(self.mat.model, TYPES.Hes_base):
            price = val1 + (val2 - val1) * (self.mat.model.v_0 - v[k0 - 1]) / (
                v[k0] - v[k0 - 1]
            )
        else:
            raise NotImplementedError
        return price

    @abstractmethod
    def _get_val(self, k0: int):
        raise NotImplementedError

    @abstractmethod
    def _init_variables(self):
        raise NotImplementedError
