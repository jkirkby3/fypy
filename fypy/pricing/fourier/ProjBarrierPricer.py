from typing import Dict, Any

import numpy as np
import scipy

from fypy.model.levy.LevyModel import FourierModel
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.pricing.fourier.ProjEuropeanPricer import LinearImpl


class ProjBarrierPricer(StrikesPricer):
    def __init__(self,
                 model: FourierModel,
                 N: int = 2 ** 11,
                 L: float = 10.,
                 order: int = 1,
                 alpha_override: float = np.nan):
        """
        Price Barrier options using the Frame Projection (BPROJ) method of Kirkby (2014)

        Ref: JL Kirkby,ROBUST BARRIER OPTION PRICING BY FRAME PROJECTION UNDER EXPONENTIAL LEVY DYNAMICS

        :param model: Fourier model
        :param N: int (power of 2), number of basis coefficients (increase to increase accuracy)
        :param L: float, controls gridwidth of density. A value of L = 10~14 works well... For Black-Scholes,
            L = 6 is fine, for heavy tailed processes such as CGMY, may want a larger value to get very high accuracy
        :param order: int, the Spline order: 0 = Haar, 1 = Linear, 2 = Quadratic, 3 = Cubic
            Note: Only Linear case is implemented until now
        :param alpha_override: float, if supplied, this overrides the rule using L to determine the gridwidth,
            allows you to use your own rule to set grid if desired
        """
        self._model = model
        self._order = order
        self._N = N
        self._L = L
        self._alpha_override = alpha_override
        self._efficient_multi_strike = [1]

    def price_strikes(self,
                      T: float = None,
                      M: float = None,
                      H: float = None,
                      down: float = None,
                      rebate: float = None,
                      K: np.ndarray = [None],
                      is_calls: np.ndarray = [None]) -> np.ndarray:
        """
        override of the method of StrikesPricer in order to handle new parameters
        Note: need to be override to handle multi-strike
        """
        output = np.empty_like(K, dtype=float)
        for i in range(len(output)):
            output_i = np.asarray([output[i]])
            self.price_strikes_fill(T=T, K=np.asarray([K[i]]), is_calls=np.asarray([is_calls[i]]), output=output_i, M=M,
                                    H=H, down=down, rebate=rebate)
            output[i] = output_i
        return output

    def price(self,
              T: float = None,
              M: float = None,
              H: float = None,
              down: float = None,
              rebate: float = None,
              K: float = None,
              is_call: bool = None) -> float:
        """
        override of the method of StrikesPricer in order to handle new parameters
        """
        prices = self.price_strikes(T=T, K=np.asarray([K]), is_calls=np.asarray([is_call]), M=M, H=H, down=down,
                                    rebate=rebate)
        return prices[0]

    def price_strikes_fill(self,
                           T: float = None,
                           M: float = None,
                           H: float = None,
                           down: float = None,
                           rebate: float = None,
                           K: np.ndarray = [None],
                           is_calls: np.ndarray = [None],
                           output: np.ndarray = [None]):

        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param is_calls: np.ndarray[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :return: None, this method fills in the output array, make sure its sized properly first
        """
        # option_params is a dictionary that includes all the parameters associated with the barrier option
        option_params = {
            'S0': self._model.spot(),
            'T': T,
            'K': K,
            'nrdt': self._get_dt_values(T=T, M=M)[0],
            'nqdt': self._get_dt_values(T=T, M=M)[1],
            'M': int(M),
            'H': H,
            'rebate': rebate,
            'val_rebate': 0,
            'is_calls': is_calls
        }

        # grid_params is a dictionary that includes all the parameters associated with the numerical grid
        grid_params = {
            'dx': self._get_dx(T=T, K=K),
            'nbar': self._nbar_computation(T=T, K=K, dx=self._get_dx(T=T, K=K)),
            'nnot ': 0,
            'xmin': 0,
            'a': 1. / self._get_dx(T=T, K=K),
            'mult': 1,
            'interp_Atend': 0,
            'b3': np.sqrt(15),
            'b4': np.sqrt(15) / 10,
            'grid_K': int(self.get_N() / 2)
        }

        # thet_params is a dictionary that includes all the parameters associated with the theta arrays
        thet_params = {
            'Thet': [0] * int(grid_params['grid_K']),
            'rho': 0,
            'zeta': 0,
            'q_plus': (1 + np.sqrt(3 / 5)) / 2,
            'q_minus': (1 - np.sqrt(3 / 5)) / 2,
        }

        # DOWN AND OUT
        if down == 1:
            # DOWN AND OUT CALL
            if is_calls[0] == 1:
                self.down_and_out_call(option_params, grid_params, thet_params, output)

            # DOWN AND OUT PUT
            else:
                raise NotImplementedError("Only down and out call pricer has been implemented")


        # UP AND OUT
        else:
            # UP AND OUT CALL
            if is_calls[0] == 1:
                raise NotImplementedError("Only down and out call pricer has been implemented")

            # UP AND OUT PUT
            else:
                raise NotImplementedError("Only down and out call pricer has been implemented")

    # down_and_out_call method prices down and out call barrier options
    def down_and_out_call(self, option_params: Dict[str, Any], grid_params: Dict[str, Any],
                          thet_params: Dict[str, Any], output: np.ndarray):

        # update of numerical grid values, such as xmin, nnot, a, nbar
        self._grid_update(option_params=option_params, grid_params=grid_params)

        # computation of orthogonal projection coefficients
        beta = self._beta_computation(option_params=option_params, grid_params=grid_params)

        # update of the difference (rho) and normalized difference (zeta) with respect to the nearest grid point
        ProjBarrierPricer._rho_zeta_update(option_params=option_params, grid_params=grid_params,
                                           thet_params=thet_params)

        # update of the parameter val_rebate, depending on the option parameter "rebate"
        ProjBarrierPricer._val_rebate_update(option_params=option_params, grid_params=grid_params, beta=beta)

        # computation of basic integrals related to payoff coefficients
        varthet_01, varthet_star = ProjBarrierPricer._payoff_constants_computation(grid_params=grid_params)

        # update of payoff integrals (Theta)
        ProjBarrierPricer._Thet_update(option_params=option_params, grid_params=grid_params, thet_params=thet_params,
                                       varthet_01=varthet_01, varthet_star=varthet_star)

        Val = ProjBarrierPricer._Val_computation(option_params=option_params, grid_params=grid_params,
                                                 thet_params=thet_params,
                                                 beta=beta, varthet_star=varthet_star)

        ProjBarrierPricer._price_doc_computation(grid_params=grid_params, Val=Val, output=output)

    def _grid_update(self, option_params: Dict[str, Any], grid_params: Dict[str, Any]):

        # The method _grid_update adjusts the parameters of the grid used
        # It modifies the grid based on the barrier and initial asset prices,
        # and handles special cases such as down-and-out call options where the barrier price is near the initial asset price.
        # It also sets the grid spacing and the scaling factor for the grid

        l = np.log(option_params['H'] / option_params['S0'])
        grid_params['xmin'] = l
        grid_params['nnot'] = int(np.floor(1 - grid_params['xmin'] * grid_params['a']))

        if grid_params['nnot'] >= grid_params['grid_K']:
            print(f"nnot is {grid_params['nnot']} while grid_K is {grid_params['grid_K']}, need to increase alpha")

        if option_params['is_calls'][
            0] == 1 and grid_params[
            'nnot'] == 1:  # In this case a DOC with H near S0 is still very valuable, so setting alph too small is bad idea
            grid_params['interp_Atend'] = 1  # Instruct to use interpolation at algorithm end
            # no change is made to dx
        else:
            nnot = max(2, int(np.floor(1 - grid_params['xmin'] * grid_params['a'])))
            grid_params['dx'] = l / (1 - nnot)

        grid_params['a'] = 1 / grid_params['dx']
        grid_params['nbar'] = self.get_nbar(a=grid_params['a'], lws=np.log(option_params['K'][0] / option_params['S0']),
                                            lam=grid_params['xmin'])

    def _beta_computation(self, option_params: Dict[str, Any], grid_params: Dict[str, Any]):

        # the method _beta_computation computes the orthogonal projection coefficients

        zmin = (1 - grid_params['grid_K']) * grid_params['dx']  # Kbar corresponds to zero

        Nmult = grid_params['mult'] * self._N
        Cons = 24 * (grid_params['a'] ** 2) * np.exp(option_params['nrdt']) / Nmult

        impl = LinearImpl(N=self._N, dx=grid_params['dx'], model=self._model, T=option_params['T'] / option_params['M'],
                          max_n_bar=grid_params['nbar'])
        return Cons * np.real(scipy.fft.fft(impl.integrand(xmin=zmin)))

    @staticmethod
    def _rho_zeta_update(option_params: Dict[str, Any], grid_params: Dict[str, Any],
                         thet_params: Dict[str, Any]):

        # update of the difference (rho) and normalized difference (zeta) with respect to the nearest grid point

        thet_params['rho'] = np.log(option_params['K'][0] / option_params['S0']) - (
                grid_params['xmin'] + (grid_params['nbar'] - 1) * grid_params['dx'])
        thet_params['zeta'] = grid_params['a'] * thet_params['rho']

    @staticmethod
    def _payoff_constants_computation(grid_params: Dict[str, Any]):

        # computation of basic integrals related to payoff coefficients

        varthet_01 = np.exp(.5 * grid_params['dx']) * (
                5 * np.cosh(grid_params['b4'] * grid_params['dx']) - grid_params['b3'] * np.sinh(
            grid_params['b4'] * grid_params['dx']) + 4) / 18
        varthet_m10 = np.exp(-.5 * grid_params['dx']) * (
                5 * np.cosh(grid_params['b4'] * grid_params['dx']) + grid_params['b3'] * np.sinh(
            grid_params['b4'] * grid_params['dx']) + 4) / 18
        varthet_star = varthet_01 + varthet_m10

        return varthet_01, varthet_star

    @staticmethod
    def _val_rebate_update(option_params: Dict[str, Any], grid_params: Dict[str, Any],
                           beta: np.ndarray):
        # if rebate is not 0option_params.
        if option_params['rebate'] != 0:
            option_params['val_rebate'] = option_params['rebate'] * np.concatenate(
                [np.flip(np.cumsum(beta[:grid_params['grid_K'] - 1])),
                 [0]])  # NOTE: this includes the discounting via beta
        else:
            option_params['val_rebate'] = 0

    @staticmethod
    def _d_computation(grid_params: Dict[str, Any], thet_params: Dict[str, Any]):

        sigma = 1 - thet_params['zeta']
        sigma_plus = (thet_params['q_plus'] - .5) * sigma
        sigma_minus = (thet_params['q_minus'] - .5) * sigma

        es1 = np.exp(grid_params['dx'] * sigma_plus)
        es2 = np.exp(grid_params['dx'] * sigma_minus)

        dbar_0 = .5 + thet_params['zeta'] * (.5 * thet_params['zeta'] - 1)
        dbar_1 = sigma * (1 - .5 * sigma)

        d_0 = np.exp((thet_params['rho'] + grid_params['dx']) * .5) * sigma ** 2 / 18 * (
                5 * ((1 - thet_params['q_minus']) * es2 + (1 - thet_params['q_plus']) * es1) + 4)
        d_1 = np.exp((thet_params['rho'] - grid_params['dx']) * .5) * sigma / 18 * (
                5 * ((.5 * (thet_params['zeta'] + 1) + sigma_minus) * es2 + (
                .5 * (thet_params['zeta'] + 1) + sigma_plus) * es1) + 4 * (
                        thet_params['zeta'] + 1))

        return d_0, dbar_0, d_1, dbar_1

    @staticmethod
    def _Thet_update(option_params: Dict[str, Any], grid_params: Dict[str, Any],
                     thet_params: Dict[str, Any], varthet_01: float, varthet_star: float):

        # update of payoff integrals (Theta)
        d_0, dbar_0, d_1, dbar_1 = ProjBarrierPricer._d_computation(grid_params=grid_params, thet_params=thet_params)
        thet_params['Thet'][grid_params['nbar'] - 1] = option_params['K'][0] * (
                np.exp(-thet_params['rho']) * d_0 - dbar_0)
        thet_params['Thet'][grid_params['nbar']] = option_params['K'][0] * (
                np.exp(grid_params['dx'] - thet_params['rho']) * (varthet_01 + d_1) - (.5 + dbar_1))
        thet_params['Thet'][grid_params['nbar'] + 1:grid_params['grid_K']] = np.exp(
            grid_params['xmin'] + grid_params['dx'] * np.arange(grid_params['nbar'] + 1,
                                                                grid_params['grid_K'])) * option_params[
                                                                                 'S0'] * varthet_star - \
                                                                             option_params['K'][0]

        thet_params['Thet'][0] = thet_params['Thet'][0] + 0.5 * option_params['rebate']
        thet_params['Thet'] = (np.array(thet_params['Thet'])).flatten()

    @staticmethod
    def _toepM_toepR_computation(grid_params: Dict[str, Any], beta: np.ndarray):
        toepM = np.concatenate(
            [beta[grid_params['grid_K'] - 1::-1], [0],
             beta[2 * grid_params['grid_K'] - 2:grid_params['grid_K'] - 1:-1]])
        toepM = scipy.fft.fft(toepM)
        toepR = np.concatenate(
            [beta[2 * grid_params['grid_K'] - 1:grid_params['grid_K'] - 1:-1], [0],
             np.zeros(grid_params['grid_K'] - 1)])
        toepR = scipy.fft.fft(toepR)

        return toepM, toepR

    @staticmethod
    def _Thetbar_computation(option_params: Dict[str, Any], grid_params: Dict[str, Any],
                             beta: np.ndarray, varthet_star: float,
                             toepR: np.ndarray):

        Thetbar1 = np.exp(-option_params['nrdt']) * option_params['K'][0] * np.cumsum(
            beta[2 * grid_params['grid_K'] - 1:grid_params['grid_K'] - 1:-1])

        Thetbar2 = np.exp(-option_params['nrdt']) * option_params['S0'] * varthet_star * np.exp(
            grid_params['xmin'] + grid_params['dx'] * np.arange(grid_params['grid_K'], 2 * grid_params['grid_K']))

        Thetbar2_fft = scipy.fft.fft(np.concatenate([Thetbar2, np.zeros(len(toepR) - len(Thetbar2))]))

        p = np.real(scipy.fft.ifft(toepR * Thetbar2_fft))

        Thetbar2 = p[:grid_params['grid_K']]

        return Thetbar1, Thetbar2

    @staticmethod
    def _p_computation(grid_params: Dict[str, Any],
                       thet_params: Dict[str, Any], toepM: np.ndarray,
                       ):

        Thet_fft = scipy.fft.fft(np.concatenate(
            [thet_params['Thet'][:grid_params['grid_K']],
             np.zeros(len(toepM) - len(thet_params['Thet'][:grid_params['grid_K']]))]))
        p = np.real(scipy.fft.ifft(toepM * Thet_fft))

        return p

    @staticmethod
    def _Val_update(option_params: Dict[str, Any], grid_params: Dict[str, Any],
                    thet_params: Dict[str, Any], toepM: np.ndarray, Thetbar1: np.ndarray, Thetbar2: np.ndarray,
                    Val: np.ndarray):

        # final for cycle of the pricing method needed to update the payoff integrals theta

        for m in range(option_params['M'] - 2, -1, -1):
            thet_params['Thet'][1:grid_params['grid_K'] - 1] = (Val[:grid_params['grid_K'] - 2] + 10 * Val[
                                                                                                       1:grid_params[
                                                                                                             'grid_K'] - 1] + Val[
                                                                                                                              2:
                                                                                                                              grid_params[
                                                                                                                                  'grid_K']]) / 12
            thet_params['Thet'][0] = (13 * Val[0] + 15 * Val[1] - 5 * Val[2] + Val[3]) / 48
            thet_params['Thet'][grid_params['grid_K'] - 1] = 2 * (
                    13 * Val[grid_params['grid_K'] - 1] + 15 * Val[grid_params['grid_K'] - 2] - 5 * Val[
                grid_params['grid_K'] - 3] +
                    Val[
                        grid_params['grid_K'] - 4]) / 48  # NOTE: 2*theta(K) b/c of augmenting

            thet_params['Thet'][0] = thet_params['Thet'][
                                         0] + 0.5 * option_params[
                                         'rebate']  # account for overhang into the knock-out region

            p = scipy.fft.ifft(
                toepM * scipy.fft.fft(
                    np.concatenate([thet_params['Thet'][:grid_params['grid_K']], np.zeros(grid_params['grid_K'])])))
            Val[:grid_params['grid_K']] = np.real(p[:grid_params['grid_K']] + np.exp(
                option_params['nqdt'] * (option_params['M'] - m - 1)) * Thetbar2 - np.exp(
                option_params['nrdt'] * (option_params['M'] - m - 1)) * Thetbar1)

            if option_params['rebate'] != 0:
                Val = Val + option_params['val_rebate']
        return Val

    @staticmethod
    def _Val_computation(option_params: Dict[str, Any], grid_params: Dict[str, Any],
                         thet_params: Dict[str, Any], beta: np.ndarray, varthet_star: float):

        toepM, toepR = ProjBarrierPricer._toepM_toepR_computation(grid_params=grid_params, beta=beta)

        Thetbar1, Thetbar2 = ProjBarrierPricer._Thetbar_computation(option_params=option_params,
                                                                    grid_params=grid_params,
                                                                    beta=beta,
                                                                    varthet_star=varthet_star,
                                                                    toepR=toepR)

        p = ProjBarrierPricer._p_computation(grid_params=grid_params, thet_params=thet_params, toepM=toepM)
        if option_params['rebate'] != 0:
            Val = p[:grid_params['grid_K']] + np.exp(option_params['nrdt']) * (Thetbar2 - Thetbar1) + option_params[
                'val_rebate']
        else:
            Val = p[:grid_params['grid_K']] + np.exp(option_params['nrdt']) * (Thetbar2 - Thetbar1)

        return ProjBarrierPricer._Val_update(option_params=option_params, grid_params=grid_params,
                                             thet_params=thet_params,
                                             toepM=toepM, Thetbar1=Thetbar1, Thetbar2=Thetbar2, Val=Val)

    @staticmethod
    def _price_doc_computation(grid_params: Dict[str, Any], Val: np.ndarray, output: np.ndarray):
        if grid_params['interp_Atend'] == 1:
            dd = 0 - (grid_params['xmin'] + (grid_params['nnot'] - 1) * grid_params['dx'])
            price = Val[grid_params['nnot'] - 1] + (
                    Val[grid_params['nnot']] - Val[
                grid_params['nnot'] - 1]) * dd / grid_params['dx']  # ie linear interp of nnot and nnot+1
            output[0] = max(0, price)


        else:
            price = Val[grid_params['nnot'] - 1]
            output[0] = max(0, price)

    def _get_dt_values(self, T: float, M: float):
        dt = T / M
        nrdt = -self.get_r(T) * dt
        nqdt = -self.get_q(T) * dt
        return nrdt, nqdt

    def _get_dx(self, T: float, K: np.ndarray):
        lws_vec = np.log(K[0] / self.get_model().spot())

        cumulants = self.get_model().cumulants(T)
        alph = cumulants.get_truncation_heuristic(L=self.get_L()) \
            if np.isnan(self.get_alpha_override()) else self.get_alpha_override()

        # Ensure that grid is wide enough to cover the strike
        alph = max(alph, 1.15 * np.max(np.abs(lws_vec)) + cumulants.c1)
        return 2 * alph / (self.get_N() - 1)

    def _nbar_computation(self, T: float, K: np.ndarray, dx: float):
        cumulants = self.get_model().cumulants(T)
        lam = cumulants.c1 - (self.get_N() / 2 - 1) * dx
        return self.get_nbar(a=1. / dx, lws=np.log(K[0] / self.get_model().spot()),
                             lam=lam)

    def get_nbar(self, a: float, lws: float, lam: float) -> int:
        try:
            nbar = int(np.floor(a * (lws - lam) + 1))
            if nbar >= self._N:
                nbar = self._N - 1
        except Exception as e:
            raise e
        return nbar

    def get_N(self):
        return self._N

    def get_model(self):
        return self._model

    def get_alpha_override(self):
        return self._alpha_override

    def get_L(self):
        return self._L

    def get_r(self, T:float):
        return self._model.discountCurve.implied_rate(T)

    def get_q(self, T:float):
        if getattr(self._model.forwardCurve, 'divDiscountCurve', None) is not None:
            return self._model.forwardCurve.divDiscountCurve.implied_rate(T)
        else:
            return 0
