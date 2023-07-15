import numpy as np

from fypy.model.levy.LevyModel import LevyModel
from fypy.pricing.fourier.ProjPricer import ProjPricer, Impl, CubicImpl


class ProjAsianPricer(ProjPricer):

    def __init__(self, model: LevyModel, N: int = 2 ** 11, L: float = 10., order: int = 3,
                 alpha_override: float = np.nan):

        super().__init__(model, N, L, order, alpha_override)

    def price_strikes(self,
                      T: float = None,
                      M: int = None,
                      K: np.ndarray = None,
                      is_calls=None) -> np.ndarray:
        """
        override of the method of StrikesPricer in order to handle new parameters
        """
        if is_calls is None:
            is_calls = [None]
        output = np.empty_like(K, dtype=float)
        self.price_strikes_fill(T=T, K=K, M=M, is_calls=is_calls, output=output)
        return output

    def price(self,
              T: float = None,
              M: int = None,
              K: float = None,
              is_call: bool = None) -> float:
        """
        override of the method of StrikesPricer in order to handle new parameters
        """
        prices = self.price_strikes(T=T, K=np.asarray([K]), is_calls=np.asarray([is_call]), M=M)
        return prices[0]

    def price_strikes_fill(self,
                           T: float = None,
                           M: int = None,
                           K: np.ndarray = None,
                           is_calls: np.ndarray = None,
                           output: np.ndarray = None):

        if len(output) != len(is_calls) or len(K) != len(is_calls):
            raise ValueError("You supply")
        for i in range(len(K)):
            output[i] = self.price(T, K[i], is_calls[i])

    def _beta_computation(self):
        raise NotImplementedError


class ProjArithmeticAsianPricer(ProjAsianPricer):

    def __init__(self,
                 model: LevyModel,
                 N: int = 2 ** 11,
                 L: float = 10.,
                 order: int = 3,
                 alpha_override: float = np.nan):
        """
        Price Asian options using the Frame Projection (APROJ) method of Kirkby (2016)
        Ref: JL Kirkby, An Efficient Transform Method for Asian Option Pricing

        :param model: Fourier model
        :param N: int (power of 2), number of basis coefficients (increase to increase accuracy)
        :param L: float, controls gridwidth of density. A value of L = 10~14 works well... For Black-Scholes,
            L = 6 is fine, for heavy tailed processes such as CGMY, may want a larger value to get very high accuracy
        :param order: int, the Spline order: 0 = Haar, 1 = Linear, 2 = Quadratic, 3 = Cubic
            Note: Only Cubic case is implemented until now
        :param alpha_override: float, if supplied, this overrides the rule using L to determine the gridwidth,
            allows you to use your own rule to set grid if desired
        """

        super(ProjArithmeticAsianPricer, self).__init__(model=model, N=N, L=L, order=order,
                                                        alpha_override=alpha_override)

    def price_strikes_fill(self,
                           T: float = None,
                           M: float = None,
                           K: np.ndarray = None,
                           is_calls: np.ndarray = None,
                           output: np.ndarray = None):

        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param M:   = number of subintervals of [0,T] (total of M+1 monitoring points in time grid, including S_0)
        :param is_calls: np.ndarray[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :return: None, this method fills in the output array, make sure its sized properly first
        """

        # definition of option_params as a structured numpy array, in order to use numba
        # the array contains: S0, T, M, dt, C
        option_params = np.array([(self.get_model().spot(),
                                   T,
                                   M,
                                   T / M,
                                   self.get_model().spot() / (M + 1))],
                                 dtype=[('S_0', float),
                                        ('T', float),
                                        ('M', int),
                                        ('dt', float),
                                        ('C', float)])

        # definition of grid_params_default as a structured numpy array, in order to use numba
        # the array contains: dx, a, alpha, ystar, nbar, AA, C_aN
        grid_params = np.zeros(1,
                               dtype=[('dx', float),
                                      ('a', float),
                                      ('alpha', float),
                                      ('ystar', float),
                                      ('nbar', int),
                                      ('AA', float),
                                      ('C_aN', float)])

        # alpha setting
        grid_params['alpha'] = self.get_alpha(T=T)

        # firs update of most of the parameters
        self._grid_update(grid_params=grid_params)

        impl = CubicImpl(N=self._N, dx=grid_params['dx'].item(), model=self._model, T=option_params['dt'].item(),
                         max_n_bar=grid_params['nbar'].item())

        # benhamou shift arrays computation
        x1, Nm = self.benhamou_shift_computation(option_params=option_params, grid_params=grid_params)

        # orthogonal projection coefficients computation
        beta = self._beta_computation(option_params=option_params, grid_params=grid_params, x1=x1,
                                      Nm=Nm, impl=impl)

        # getting payoff constants for arithmetic asian option
        payoff_coefficient_constants = ProjArithmeticAsianPricer.asian_coefficient_constants(
            C=option_params['C'].item(), dx=grid_params['dx'].item(), impl=impl)

        for index in range(len(K)):
            # adjustment of the orthogonal projection coefficients for each strike
            beta_adjusted = self._beta_strike_adjustment(option_params=option_params, grid_params=grid_params, impl=impl,
                                                        x1=x1, beta=beta, index=index, K=K)

            # computation of the payoff coefficients
            G = ProjArithmeticAsianPricer._payoff_coefficient_computation(option_params=option_params, grid_params=grid_params,
                                                       payoff_coefficient_constants=payoff_coefficient_constants,
                                                       index=index, K=K)

            # final update of the prices
            self._price_update(option_params=option_params, grid_params=grid_params, beta_adjusted=beta_adjusted, G=G,
                                 index=index,
                                 K=K, is_calls=is_calls,
                                 output=output)

    def _grid_update(self, grid_params: np.ndarray):

        grid_params['dx'] = 2 * grid_params['alpha'].item() / (self._N - 1)
        grid_params['a'] = 1 / grid_params['dx'].item()

        A = 32 * grid_params['a'].item() ** 4

        grid_params['C_aN'] = A / self._N
        grid_params['AA'] = 1 / A

    def benhamou_shift_computation(self, option_params: np.ndarray, grid_params: np.ndarray):

        # ER : risk neutral expected return over increment dt=1/M (can set to zero if it is unknown)
        ER = (self.get_r(option_params['T'].item())
              - self.get_q(option_params['T'].item())
              + self.get_model().convexity_correction()) * (option_params['dt'].item())

        M = option_params['M'].item()
        x1 = np.zeros(M)
        x1[0] = ER

        for m in range(1, M):
            x1[m] = ER + np.log(1 + np.exp(x1[m - 1]))  # BENHAMOU SHIFT

        Nm = np.floor(grid_params['a'].item() * (x1 - ER))
        x1 = ER + (1 - self._N / 2) * grid_params['dx'].item() + Nm * grid_params['dx'].item()
        return x1, Nm


    def _beta_computation(self, option_params: np.ndarray = None, grid_params: np.ndarray = None, impl: Impl = None,
                          x1: np.ndarray = None,
                          Nm: np.ndarray = None
                          ):

        xi = impl.w[1:]
        zeta = impl.zeta()

        PhiR = np.concatenate(
            ([1], np.exp((option_params['dt'].item()) * self.get_model().symbol(xi))))

        beta = np.concatenate(
            ([grid_params['AA'].item()], zeta * PhiR[1:self._N] * np.exp(-1j * x1[0] * xi)))

        N = self.get_N()
        beta = np.real(np.fft.fft(beta))

        PhiR = grid_params['C_aN'].item() * PhiR

        PSI = self._PSI_computation(M=option_params['M'].item(), dx=grid_params['dx'].item(),
                                      a=grid_params['a'].item(), x1=x1, Nm=Nm)

        # Convert both matrices to complex128 before using np.dot()
        beta = np.dot(PSI[:, :N].astype(np.complex128), beta.astype(np.complex128)) * PhiR

        for m in range(2, option_params['M'].item()):
            beta[1:N] = zeta * beta[1:N] * np.exp(-1j * x1[m - 1] * xi)
            beta[0] = grid_params['AA'].item()
            beta = np.real(np.fft.fft(beta))
            beta = np.dot(PSI[:, int(Nm[m - 1]):int(Nm[m - 1]) + N].astype(np.complex128),
                          beta.astype(np.complex128)) * PhiR
        return beta


    @staticmethod
    def asian_coefficient_constants(C: float, dx: float, impl: Impl):

        # getting the used constants through impl
        constants = np.asarray([(1 / 24 - impl.g1) * np.exp(-dx), 0.5 - impl.g2,
                                (23 / 24 - impl.g3) * np.exp(dx), impl.g4 / 90])

        return C * constants

    @staticmethod
    def _nbar_update(option_params: np.ndarray, grid_params: np.ndarray, x1: np.ndarray, index: int, K: np.ndarray):

        grid_params['ystar'] = np.log((option_params['M'].item() + 1) * K[index] / option_params['S_0'].item() - 1)
        grid_params['nbar'] = int(
            np.floor((grid_params['ystar'].item() - x1[option_params['M'].item() - 1]) * grid_params['a'] + 1))

    def _grid_widening(self, option_params: np.ndarray, grid_params: np.ndarray, K: np.ndarray):
        # this method is used iff the grid is not wide enough

        # first of all, the alpha is set
        grid_params['alpha'] = max(1.15 * grid_params['alpha'].item(), 1.15 * np.max(
            np.abs(np.log(K / self.get_model().spot()))) + self.get_model().cumulants(option_params['T'].item()).c1)

        # the grid is re-updated accordingly
        self._grid_update(grid_params=grid_params)

        # all the necessary values are re-computed
        impl = CubicImpl(N=self._N, dx=grid_params['dx'].item(), model=self._model, T=option_params['dt'].item(),
                         max_n_bar=grid_params['nbar'].item())
        x1, Nm = self.benhamou_shift_computation(option_params=option_params, grid_params=grid_params)
        beta = self._beta_computation(option_params=option_params, grid_params=grid_params, impl=impl, x1=x1,
                                      Nm=Nm)

        return impl, x1, beta

    def _beta_strike_adjustment(self, option_params: np.ndarray, grid_params: np.ndarray, impl: Impl, x1: np.ndarray,
                               beta: np.ndarray,
                               index: int, K: np.ndarray):

        ProjArithmeticAsianPricer._nbar_update(option_params=option_params, grid_params=grid_params, x1=x1, index=index,
                                               K=K)

        # this while is used iff the grid is not wide enough
        while grid_params['nbar'].item() + 1 > self._N:
            impl, x1, beta = self._grid_widening(option_params=option_params,
                                                grid_params=grid_params, K=K)

        x1_adjusted, beta_adjusted = ProjAsianPricer.copy_original_arrays(x1, beta)

        # shift adjustment
        x1_adjusted[option_params['M'].item() - 1] = grid_params['ystar'].item() - (grid_params['nbar'].item() - 1) * grid_params['dx'].item()

        # beta adjustment
        beta_adjusted[1:self._N] = impl.zeta() * beta_adjusted[1:self._N] * np.exp(
            -1j * x1_adjusted[option_params['M'].item() - 1] * impl.w[1:])

        beta_adjusted[0] = grid_params['AA'].item()

        return np.real(np.fft.fft(beta_adjusted))

    @staticmethod
    def _thet_computation(dx: float, x1: np.ndarray, Neta: int, Neta5: int):
        g2 = np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 6
        g3 = np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 6

        thet = np.zeros(Neta)
        thet[5 * np.arange(1, Neta5 + 1) - 3] = x1[0] - 1.5 * dx + dx * np.arange(Neta5)
        thet[5 * np.arange(1, Neta5 + 1) - 5] = x1[0] - 1.5 * dx + dx * np.arange(Neta5) - dx * g3
        thet[5 * np.arange(1, Neta5 + 1) - 4] = x1[0] - 1.5 * dx + dx * np.arange(Neta5) - dx * g2
        thet[5 * np.arange(1, Neta5 + 1) - 2] = x1[0] - 1.5 * dx + dx * np.arange(Neta5) + dx * g2
        thet[5 * np.arange(1, Neta5 + 1) - 1] = x1[0] - 1.5 * dx + dx * np.arange(Neta5) + dx * g3

        return thet

    @staticmethod
    def _sig_computation():
        # Weights
        g2 = np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 6
        g3 = np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 6
        v1 = .5 * 128 / 225
        v2 = .5 * (322 + 13 * np.sqrt(70)) / 900
        v3 = .5 * (322 - 13 * np.sqrt(70)) / 900

        sig = np.array(
            [-1.5 - g3, -1.5 - g2, -1.5, -1.5 + g2, -1.5 + g3, -.5 - g3, -.5 - g2, -.5, -.5 + g2, -.5 + g3])
        sig[0:5] = (sig[0:5] + 2) ** 3 / 6
        sig[5:10] = 2 / 3 - .5 * sig[5:10] ** 3 - sig[5:10] ** 2

        sig[np.array([0, 4, 5, 9])] *= v3
        sig[np.array([1, 3, 6, 8])] *= v2
        sig[np.array([2, 7])] *= v1

        return sig

    def _PSI_computation(self, M: int, dx:float, a:float, x1: np.ndarray, Nm: np.ndarray):
        # PSI Matrix: 5-Point GAUSSIAN
        N = self.get_N()
        NNM = int(N + Nm[M - 2])  # Number of columns of PSI
        Neta = 5 * NNM + 15  # Sample size
        Neta5 = NNM + 3
        thet = ProjArithmeticAsianPricer._thet_computation(dx=dx, x1=x1, Neta=Neta, Neta5=Neta5)
        sig = ProjArithmeticAsianPricer._sig_computation()
        dxi = 2 * np.pi * a / N
        zz = np.exp(1j * dxi * np.log(1 + np.exp(thet)))
        thet = zz

        PSI = np.zeros((N, NNM), dtype=np.float64)  # The first row will remain ones
        PSI = PSI.astype(np.complex128)
        PSI[0, :] = np.ones(NNM)

        for j in range(1, N - 1):
            PSI[j, :] = (sig[0] * (thet[0:Neta - 19:5] + thet[19:Neta:5])
                         + sig[1] * (thet[1:Neta - 18:5] + thet[18:Neta - 1:5])
                         + sig[2] * (thet[2:Neta - 17:5] + thet[17:Neta - 2:5])
                         + sig[3] * (thet[3:Neta - 16:5] + thet[16:Neta - 3:5])
                         + sig[4] * (thet[4:Neta - 15:5] + thet[15:Neta - 4:5])
                         + sig[5] * (thet[5:Neta - 14:5] + thet[14:Neta - 5:5])
                         + sig[6] * (thet[6:Neta - 13:5] + thet[13:Neta - 6:5])
                         + sig[7] * (thet[7:Neta - 12:5] + thet[12:Neta - 7:5])
                         + sig[8] * (thet[8:Neta - 11:5] + thet[11:Neta - 8:5])
                         + sig[9] * (thet[9:Neta - 10:5] + thet[10:Neta - 9:5]))

            thet = thet * zz

        return PSI



    @staticmethod
    def _payoff_coefficient_computation(option_params: np.ndarray, grid_params: np.ndarray,
                                          payoff_coefficient_constants: np.ndarray, index: int, K: np.ndarray):
        dx = grid_params['dx'].item()
        nbar = grid_params['nbar'].item()

        G = np.zeros(nbar + 1)
        E = np.exp(grid_params['ystar'].item() - (nbar - 1) * dx + dx * np.arange(nbar + 1))
        D = K[index] - option_params['C'].item()

        G[nbar] = D / 24 - payoff_coefficient_constants[0] * E[nbar]
        G[nbar - 1] = .5 * D - payoff_coefficient_constants[1] * E[nbar - 1]
        G[nbar - 2] = 23 * D / 24 - payoff_coefficient_constants[2] * E[nbar - 2]
        G[0:nbar - 2] = D - payoff_coefficient_constants[3] * E[0:nbar - 2]

        return G

    def _price_update(self, option_params: np.ndarray, grid_params: np.ndarray, beta_adjusted: np.ndarray,
                        G: np.ndarray, index: int, K: np.ndarray, is_calls: np.ndarray,
                        output: np.ndarray):

        T = option_params['T'].item()
        M = option_params['M'].item()

        r = self.get_r(T)
        q = self.get_q(T)

        Val = grid_params['C_aN'].item() * np.exp(-r * T) * np.sum(beta_adjusted[0:grid_params['nbar'].item() + 1] * G)

        if is_calls[index]:  # Call Option, use Put-Call-Parity
            if r - q == 0:
                mult = M + 1
            else:
                mult = (np.exp((r - q) * T * (1 + 1 / M)) - 1) / (
                        np.exp((r - q) * option_params['dt'].item()) - 1)
            Val = Val + option_params['C'].item() * np.exp(-r * T) * mult - K[index] * np.exp(-r * T)

        output[index] = max(0, Val)
