import numpy as np
from scipy.fft import fft
from functools import partial
from fypy.model.levy.LevyModel import FourierModel
from fypy.pricing.fourier.ProjPricer import ProjPricer
from fypy.model.levy.LevyModel import LevyModel
from fypy.pricing.fourier.ProjPricer import (
    ProjPricer,
    Impl,
    CubicImpl,
    LinearImpl,
    HaarImpl,
)


class ProjForwardStartingOption(ProjPricer):
    def __init__(
        self,
        model: LevyModel,
        N: int = 2**14,
        L: float = 10.0,
        order: int = 3,
        alpha_override: float = np.nan,
    ):
        """
        Price Forward Starting options using the Frame Projection (PROJ) method of Kirkby (2015)

        Ref:ROBUST OPTION PRICING WITH CHARACTERISTIC FUNCTIONS AND THE B-SPLINE ORDER OF DENSITY PROJECTION

        :param model: Fourier model
        :param N: int (power of 2), number of basis coefficients (increase to increase accuracy)
        :param L: float, controls gridwidth of density. A value of L = 10~14 works well... For Black-Scholes,
            L = 6 is fine, for heavy tailed processes such as CGMY, may want a larger value to get very high accuracy
        :param order: int, the Spline order: 0 = Haar, 1 = Linear, 2 = Quadratic, 3 = Cubic
            Note: Cubic is preferred, the others are provided for research purposes. Only 1 and 3 are currently coded
        :param alpha_override: float, if supplied, this overrides the rule using L to determine the gridwidth,
            allows you to use your own rule to set grid if desired
        """
        super().__init__(model, N, L, order, alpha_override)
        # self._efficient_multi_strike = [1]
        self.model = model
        self._N = N
        self._alpha_override = alpha_override
        self._L = L
        if order not in [0, 1, 3]:
            raise NotImplementedError("Only linear, Haar and cubic implemented so far")
        return

    def price_strikes_fill(
        self,
        start_date: float,
        tau: float,
        S_0: float,
        is_call: bool = True,
    ) -> float:
        """Method to price a forward starting option.

        Args:
            start_date (float): start date of the forward starting option in year.
            tau (float): living time of the option
            S_0 (float): value of the underlying at t=0.
            is_call (bool, optional): if call:true, otherwise False

        Returns:
            float: price of the option with these characteristics.
        """
        # required variables to compute the price
        final_expiry = start_date + tau  # T:paydate
        div_rate = self.get_q(final_expiry)
        # compute the put value
        put_value = S_0 * self._compute_put_value_unit(
            final_expiry=final_expiry,
            start_date=start_date,
            nbar=self.get_nbar(),
            dx=self.get_dx(final_expiry=final_expiry),
        )
        # pcp formula to get call prices
        match is_call:
            case 1:
                return self._call_from_put(
                    put_value, tau, S_0, div_rate, final_expiry, start_date
                )
            case 0:
                return put_value

    def get_dx(self, final_expiry: float) -> float:
        alpha = self.get_alpha(T=final_expiry)
        return 2 * alpha / (self._N - 1)

    def get_nbar(self) -> int:
        return int(self._N / 2)

    # pcp formula
    def _call_from_put(
        self,
        put_value: float,
        tau: float,
        S_0: float,
        div_rate: float,
        final_expiry: float,
        start_date: float,
    ) -> float:
        return (
            S_0
            * (
                np.exp(-div_rate * final_expiry)
                - self.model.discountCurve(tau) * np.exp(-div_rate * start_date)
            )
            + put_value
        )

    # computes the value of
    def _compute_put_value_unit(
        self,
        dx: float,
        final_expiry: float,
        start_date: float,
        nbar: int,
    ) -> float:
        xmin = (1 - self._N / 2) * dx
        impl_t1 = self._get_impl(dx, start_date, nbar)
        impl_t2 = self._get_impl(dx, final_expiry - start_date, nbar)
        vbar2 = self._compute_vbar2(xmin, nbar, impl_t2)
        vbar1 = self._compute_vbar1(xmin, impl_t1, dx)
        vtstar = self._get_vtstar(impl_t2)
        return self.model.discountCurve(final_expiry) * vbar2 * vbar1 * vtstar

    # Compute gamma_tau = upsilon(tau)*<beta,g(1,1)>, first step described in the paper
    # essentially computes the price of a European option with maturity tau, S0 = 1
    def _compute_vbar2(self, xmin: float, nbar: int, impl: Impl) -> float:
        D_tau = impl.integrand(xmin)
        beta_tau = self._beta_computation(D_tau)
        g_tau = impl.coefficients(W=1, S0=1, xmin=xmin, nbar=nbar)
        max_idx = impl.num_coeffs(nbar)
        return impl.cons() / self._N * (g_tau[:max_idx] @ beta_tau[:max_idx].T)

    # Compute gamma = upsilon(t*)*<beta,exp>
    def _compute_vbar1(self, xmin: float, impl_t2: Impl, dx: float) -> float:
        D_tstar = impl_t2.integrand(xmin)
        beta_tstar = self._beta_computation(D_tstar)
        g_tstar = np.exp(xmin + np.arange(self._N) * dx)
        vbar1 = impl_t2.cons() / self._N * (g_tstar @ beta_tstar.T)
        return vbar1

    # compute the beta vector, as defined in the paper
    def _beta_computation(self, D: np.array) -> np.ndarray:
        return np.real(fft(D))

    # get the vtstar coefficient, as defined in Table 2 of the paper
    def _get_vtstar(self, impl_t2: Impl) -> float:
        match self._order:
            case 0:
                return 1
            case 1:
                return impl_t2.g2
            case 3:
                return impl_t2.g4 / 90

    # get splines base depending on the order inputted
    def _get_impl(self, dx: float, T: float, nbar: int) -> Impl:
        match self._order:
            case 0:
                return HaarImpl(
                    N=self._N, dx=dx, model=self._model, T=T, max_n_bar=nbar
                )
            case 1:
                return LinearImpl(
                    N=self._N, dx=dx, model=self._model, T=T, max_n_bar=nbar
                )
            case 3:
                return CubicImpl(
                    N=self._N, dx=dx, model=self._model, T=T, max_n_bar=nbar
                )
