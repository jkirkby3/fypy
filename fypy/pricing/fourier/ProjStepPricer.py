import math
from typing import Dict, Any

import numpy as np
import torch
from numba import prange

from fypy.model.levy.LevyModel import LevyModel
from fypy.pricing.fourier.ProjPricer import ProjPricer, LinearImpl



class ProjStepPricer(ProjPricer):
    def __init__(self, model: LevyModel, N: int = 2 ** 10, L: float = 10., order: int = 1,
                 alpha_override: float = np.nan, alphMult: float = 1.1, TOLProb: float = 5e-08,
                 TOLMean: float = 1e-05, cuda: bool = False):
        """
        Price Step options using the Frame Projection (BPROJ) method of Kirkby (2014)

        Ref: JL Kirkby,ROBUST BARRIER OPTION PRICING BY FRAME PROJECTION UNDER EXPONENTIAL LEVY DYNAMICS

        :param model: Fourier model
        :param N: int (power of 2), number of basis coefficients (increase to increase accuracy)
        :param L: float, controls gridwidth of density. A value of L = 10~14 works well... For Black-Scholes,
            L = 6 is fine, for heavy tailed processes such as CGMY, may want a larger value to get very high accuracy
        :param order: int, the Spline order: 0 = Haar, 1 = Linear, 2 = Quadratic, 3 = Cubic
            Note: Only Linear case is implemented until now
        :param alpha_override: float, if supplied, this overrides the rule using L to determine the gridwidth,
            allows you to use your own rule to set grid if desired
        :param alphMult, TOLProb, TOLMean: float parameters used to adjust grid width
        :cuda : bool, if True the GPU is used
        """
        super().__init__(model, N, L, order, alpha_override)
        self._model = model
        self._alphMult = alphMult
        self._TOLProb = TOLProb
        self._TOLMean = TOLMean
        self._cuda = cuda
        self._original_N = N

    def price_strikes(self,
                      T: float = None,
                      M: float = None,
                      H: float = None,
                      down: bool = None,
                      stepRho: float = None,
                      K: np.ndarray = None,
                      is_calls: np.ndarray = None) -> np.ndarray:
        """
        override of the method of StrikesPricer in order to handle new parameters

        """
        output = np.empty_like(K, dtype=float)
        self.price_strikes_fill(T=T, K=K, M=M, H=H, down=down,
                                stepRho=stepRho, is_calls=is_calls, output=output)
        return output

    def price(self,
              T: float = None,
              M: float = None,
              H: float = None,
              down: bool = None,
              stepRho: float = None,
              K: float = None,
              is_call: bool = None) -> float:
        """
        override of the method of StrikesPricer in order to handle new parameters
        """
        prices = self.price_strikes(T=T, K=np.asarray([K]), is_calls=np.asarray([is_call]), M=M, H=H, down=down,
                                    stepRho=stepRho)
        return prices[0]

    def price_strikes_fill(self,
                           T: float = None,
                           M: float = None,
                           H: float = None,
                           down: bool = None,
                           stepRho: float = None,
                           K: np.ndarray = None,
                           is_calls: np.ndarray = None,
                           output: np.ndarray = None):

        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param M : number of subintervals of [0,T] (total of M+1 monitoring points in time grid, including S_0)
        :param down : True for down and out (otherwise it's up and out)
        :param H  : barrier
        :param is_calls: np.ndarray[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :param stepRho : knock-out rate
        :return: None, this method fills in the output array, make sure its sized properly first
        # """

        # For Automated Parameter adjustment

        option_dict = {'S0': self._model.spot(),
                       'K': K,
                       'T': T,
                       'M': M,
                       'down': down,
                       'H': H,
                       'h': np.log(H / self._model.spot()),
                       'stepRho': stepRho}

        grid_dict = {'alph': None,
                     'dx': None,
                     'a': None,
                     'xmin': None,
                     'K_grid': None,
                     'n_h': None,
                     'nnot': None,
                     'Nmax': max(2 ** 8, 2 * self._original_N),
                     'interp_Atend': 0}

        payoff_constants_dict = {'b3': np.sqrt(15),
                                 'b4': np.sqrt(15) / 10,
                                 'varthet_01': None,
                                 'varthet_star': None,
                                 'varthet_m10': None}

        self._grid_initialization(option_dict=option_dict, grid_dict=grid_dict)

        beta = self._iterative_beta(option_dict=option_dict, grid_dict=grid_dict,
                                    payoff_constants_dict=payoff_constants_dict)

        ProjStepPricer._payoff_constants_update(grid_dict=grid_dict, payoff_constants_dict=payoff_constants_dict)

        #   DETERMINE COMMON Params

        # DOWN AND OUT
        if down:
            # DOWN AND OUT CALL
            if is_calls[0]:
                raise NotImplementedError("Only up and out call pricer has been implemented")


            # DOWN AND OUT PUT
            else:
                raise NotImplementedError("Only up and out call pricer has been implemented")

        # UP AND OUT
        else:
            # UP AND OUT CALL
            if is_calls[0]:
                ProjStepPricer._up_and_out_call(option_dict=option_dict, grid_dict=grid_dict,
                                                payoff_constants_dict=payoff_constants_dict, beta=beta, output=output,
                                                cuda=self._cuda)


            # UP AND OUT PUT
            else:
                raise NotImplementedError("Only up and out call pricer has been implemented")

    def get_alpha(self, T: float = None, h: float = None):
        if not np.isnan(self.get_alpha_override()):
            return self.get_alpha_override()

        # TODO: use get_truncation_heuristic, once you have verified Bilateral Gamma Motion's 2nd cumulant
        else:
            cumulants = self.get_model().cumulants(T=1)
            return max(2 * self._alphMult * abs(h),
                       self._L * np.sqrt(abs(cumulants.c2 * T) + np.sqrt(abs(cumulants.c4 * T))))

    def _grid_initialization(self, option_dict: Dict[str, Any], grid_dict: Dict[str, Any]):
        grid_dict['alph'] = self.get_alpha(T=option_dict['T'], h=option_dict['h'])

        # Step first: Satisfy Probability Tolerance
        ErrProb = 10
        numTimesInLoop = 0
        self._N = int(self._original_N / 2)
        grid_dict['alph'] = grid_dict['alph'] / self._alphMult  # b/c we immediately double

        while abs(ErrProb) > self._TOLProb and self._N < grid_dict['Nmax'] / 2:
            # NOTE: strictly less here b/c we have to enter into the mean error loop at least once

            grid_dict['alph'] = self._alphMult * grid_dict['alph']
            self._N = 2 * self._N
            numTimesInLoop += 1

            grid_dict['dx'] = 2 * grid_dict['alph'] / (self._N - 1)
            grid_dict['a'] = 1 / grid_dict['dx']
            grid_dict['xmin'] = -grid_dict['alph'] / 2
            ErrProb = self._ErrProb_computation(option_dict=option_dict, grid_dict=grid_dict)

    @staticmethod
    def _grid_barrier_realignment(option_dict: Dict[str, Any], grid_dict: Dict[str, Any]):
        if option_dict['h'] != 0:
            # Realign so that h and 0 are both members of grid (if possible)

            grid_dict['nnot'] = math.floor(1 - grid_dict['xmin'] / grid_dict['dx'])
            if abs(option_dict['h']) > grid_dict['dx']:
                # so that n_h != nnot

                grid_dict['dx'] = (option_dict['h'] - 0) / (grid_dict['n_h'] - grid_dict['nnot'])
                grid_dict['xmin'] = grid_dict['dx'] * (1 - grid_dict['nnot'])  # hence nnot should remain on the grid
                grid_dict['n_h'] = math.floor(
                    grid_dict['nnot'] + option_dict['h'] / grid_dict['dx'])  # Numerically Stable
        else:
            grid_dict['nnot'] = grid_dict['n_h']

    def _grid_update(self, option_dict: Dict[str, Any], grid_dict: Dict[str, Any]):
        grid_dict['dx'] = 2 * grid_dict['alph'] / (self._N - 1)

        grid_dict['xmin'] = -grid_dict['alph'] / 2
        grid_dict['n_h'] = math.floor((option_dict['h'] - grid_dict['xmin']) / grid_dict['dx'] + 1)
        grid_dict['xmin'] = option_dict['h'] - (grid_dict['n_h'] - 1) * grid_dict['dx']

        ProjStepPricer._grid_barrier_realignment(option_dict=option_dict, grid_dict=grid_dict)

        grid_dict['a'] = 1 / grid_dict['dx']

    def _ErrProb_computation(self, option_dict: Dict[str, Any], grid_dict: Dict[str, Any]):
        dw = 2 * np.pi * grid_dict['a'] / self._N
        xmax = grid_dict['xmin'] + (self._N / 2 - 1) * grid_dict['dx']

        gam1 = (xmax - grid_dict['xmin']) / 2
        gam2 = (xmax + grid_dict['xmin']) / 2

        grand = dw * np.arange(1, self._N)

        Prob = np.sum(np.exp(-1j * gam2 * grand) * np.sin(gam1 * grand) / grand * np.exp(
            option_dict['T'] * self._model.symbol(grand)))
        Prob += np.sum(np.exp(1j * gam2 * grand) * np.sin(gam1 * grand) / grand * np.exp(
            option_dict['T'] * self._model.symbol(-grand)))

        Prob = dw / np.pi * (gam1 + Prob)

        return 1 - Prob

    def _get_dt_values(self, T: float, M: float):
        dt = T / M
        nrdt = -self.get_r(T) * dt
        n_rq_dt = -(self.get_r(T) - self.get_q(T)) * dt
        return nrdt, n_rq_dt

    def _beta_computation(self, option_dict: Dict[str, Any] = None, grid_dict: Dict[str, Any] = None,
                          zmin: float = None):
        Cons2 = 24 * (grid_dict['a'] ** 2) * np.exp(
            self._get_dt_values(T=option_dict['T'], M=option_dict['M'])[0]) / self._N

        impl = LinearImpl(N=self._N, dx=grid_dict['dx'].item(), model=self._model,
                          T=option_dict['T'] / option_dict['M'],
                          max_n_bar=0)

        beta = Cons2 * np.real(np.fft.fft(impl.integrand(xmin=zmin)))

        return beta

    def _ErrMean_computation(self, beta: np.ndarray, option_dict: Dict[str, Any], grid_dict: Dict[str, Any],
                             payoff_constants_dict: Dict[str, Any], zmin: float):
        varthet_star = (np.exp(.5 * grid_dict['dx'])
                        * (5 * np.cosh(payoff_constants_dict['b4'] * grid_dict['dx'])
                           - payoff_constants_dict['b3'] * np.sinh(payoff_constants_dict['b4'] * grid_dict['dx']) + 4)
                        + np.exp(-.5 * grid_dict['dx'])
                        * (5 * np.cosh(payoff_constants_dict['b4'] * grid_dict['dx']) + payoff_constants_dict['b3']
                           * np.sinh(payoff_constants_dict['b4'] * grid_dict['dx']) + 4)) / 18

        ErrMean = np.exp(- self._get_dt_values(T=option_dict['T'], M=option_dict['M'])[0]) \
                  * varthet_star * np.dot(beta,np.exp( zmin + grid_dict['dx'] * np.arange( self._N))) \
                  - np.exp((-self._get_dt_values(T=option_dict['T'], M=option_dict['M'])[1]))
        # note: multiplying by exp(r*dt) to cancel the exp(-r*dt) in Cons2

        return ErrMean * option_dict['M'] * option_dict['S0']

    def _final_parameters_update(self, grid_dict: Dict[str, Any], option_dict: Dict[str, Any]):
        grid_dict['K_grid'] = int(self._N / 2)

        grid_dict['interp_Atend'] = 0
        if 0 < abs(option_dict['h']) < grid_dict['dx']:
            grid_dict['interp_Atend'] = 1

    def _iterative_beta(self, option_dict: Dict[str, Any], grid_dict: Dict[str, Any],
                        payoff_constants_dict: Dict[str, Any]):
        # Gaussian Quad Constants
        numTimesInLoop = 0
        ErrMean = 10
        # because of the grid and parameters' settings, we enter the while cycle at least ones
        while abs(ErrMean) > self._TOLMean and self._N <= grid_dict['Nmax']:
            if numTimesInLoop > 0:
                self._N = 2 * self._N  # Only double N
            numTimesInLoop += 1

            self._grid_update(option_dict=option_dict, grid_dict=grid_dict)

            zmin = (1 - self._N / 2) * grid_dict['dx']  # Kbar corresponds to zero

            beta = self._beta_computation(option_dict=option_dict, grid_dict=grid_dict, zmin=zmin)

            ErrMean = self._ErrMean_computation(option_dict=option_dict, grid_dict=grid_dict,
                                                payoff_constants_dict=payoff_constants_dict, zmin=zmin, beta=beta)

        self._final_parameters_update(option_dict=option_dict, grid_dict=grid_dict)

        return beta

    @staticmethod
    def _payoff_constants_update(grid_dict: Dict[str, Any], payoff_constants_dict: Dict[str, Any]):
        # PAYOFF CONSTANTS
        payoff_constants_dict['varthet_01'] = np.exp(.5 * grid_dict['dx']) \
                                              * (5 * np.cosh(payoff_constants_dict['b4'] * grid_dict['dx'])
                                                 - payoff_constants_dict['b3'] * np.sinh(payoff_constants_dict['b4']
                                                                                         * grid_dict['dx']) + 4) / 18
        payoff_constants_dict['varthet_m10'] = np.exp(-.5 * grid_dict['dx']) \
                                               * (5 * np.cosh(payoff_constants_dict['b4'] * grid_dict['dx'])
                                                  + payoff_constants_dict['b3'] * np.sinh(payoff_constants_dict['b4']
                                                                                          * grid_dict['dx']) + 4) / 18

        payoff_constants_dict['varthet_star'] = payoff_constants_dict['varthet_01'] \
                                                + payoff_constants_dict['varthet_m10']

    @staticmethod
    def get_nbar(S0: float = None, K: float = None, xmin: float = None, dx: float = None, a: float = None) -> int:
        return int(a * (math.log(K / S0) - xmin) + 1)

    @staticmethod
    def _d_computation(dx: float, a: float, rho: float):
        zeta = a * rho
        sigma = 1 - zeta

        q_plus = (1 + np.sqrt(3 / 5)) / 2
        q_minus = (1 - np.sqrt(3 / 5)) / 2
        # TODO: use this method also in ProjBarrierPricer
        sigma_plus = (q_plus - .5) * sigma
        sigma_minus = (q_minus - .5) * sigma

        es1 = math.exp(dx * sigma_plus)
        es2 = math.exp(dx * sigma_minus)

        dbar_0 = .5 + zeta * (.5 * zeta - 1)
        dbar_1 = sigma * (1 - .5 * sigma)
        d_0 = math.exp((rho + dx) * .5) * sigma ** 2 / 18 * (5 * ((1 - q_minus) * es2 + (1 - q_plus) * es1) + 4)
        d_1 = math.exp((rho - dx) * .5) * sigma / 18 \
              * (5 * ((.5 * (zeta + 1) + sigma_minus) * es2 + (.5 * (zeta + 1) + sigma_plus) * es1) + 4 * (zeta + 1))

        return dbar_0, dbar_1, d_0, d_1

    @staticmethod
    def _Thet_initialization(Thet: torch.Tensor, S0: float, K: float, xmin: float, dx: float, a: float, K_grid: int,
                             varthet_01: float, varthet_star: float):
        nbar = ProjStepPricer.get_nbar(S0=S0, K=K, xmin=xmin, dx=dx, a=a)

        rho = math.log(K / S0) - (xmin + (nbar - 1) * dx)

        dbar_0, dbar_1, d_0, d_1 = ProjStepPricer._d_computation(dx=dx, a=a, rho=rho)
        Thet[nbar - 1] = K * (math.exp(-rho) * d_0 - dbar_0)
        Thet[nbar] = K * (math.exp(dx - rho) * (varthet_01 + d_1) - (.5 + dbar_1))

        xmin = torch.tensor(xmin, dtype=torch.float64)

        Thet[nbar + 1:K_grid] = torch.exp(
            xmin + dx * torch.arange(nbar + 1, K_grid, dtype=torch.float64)) * S0 * varthet_star - K

    @staticmethod
    def _stepSoftener(stepRho: float, M: int, device: str):
        Gamm = M + 1
        tauM = 1 / (M + 1)
        if stepRho >= 0:  # Step Option  # NOTE: stepRho == 0 corresponds to vanilla option
            stepSoftenerArray = torch.exp(
                -stepRho * tauM * torch.arange(0, Gamm + 1, dtype=torch.float64, device=device))
        elif stepRho == -1:  # Fader Option  (Fade-in)
            stepSoftenerArray = 1 - (torch.arange(0, Gamm + 1, dtype=torch.float64, device=device) / (M + 1))
        elif stepRho == -2:  # Ordinary Barrier option (no excursion forgiveness)
            stepSoftenerArray = torch.zeros(Gamm + 1, dtype=torch.float64, device=device)
            stepSoftenerArray[0] = 1
        return stepSoftenerArray

    @staticmethod
    def _Val_initialization(stepSoftenerArray: torch.Tensor,
                            toepM: torch.Tensor,
                            Thet: torch.Tensor, Thet_vectorized: torch.Tensor, p: torch.Tensor, Val: torch.Tensor,
                            K_grid: int,  n_u: float, M: int):
        Gamm = M + 1

        Thet_vectorized[:, :n_u - 1] = (stepSoftenerArray[:Gamm, None]) * Thet[:n_u - 1]
        Thet_vectorized[:, n_u - 1] = .5 * ((stepSoftenerArray[:Gamm]) + (stepSoftenerArray[1:])) * Thet[n_u - 1]
        Thet_vectorized[:, n_u:K_grid] = (stepSoftenerArray[1:, None]) * Thet[n_u:K_grid]

        p[:] = (torch.fft.ifft(toepM * torch.fft.fft(Thet_vectorized, dim=1), dim=1)).T.real

        # Assign the real part of p to Val
        Val[:, :Gamm] = p[:K_grid, :]

    @staticmethod
    def _Val_recursion(toepM: torch.Tensor,   p: torch.Tensor, final_Thet_Val: torch.Tensor, Val: torch.Tensor,
                       M: int, K_grid: int, n_u: float, ):
        Gamm = M + 1
        for m in range(M - 2, -1, -1):
            final_Thet_Val[0, :Gamm] = (13 * Val[0, :Gamm] + 15 * Val[1, :Gamm] - 5 * Val[2, :Gamm] + Val[3,
                                                                                                      :Gamm]) / 48

            final_Thet_Val[1:n_u - 1, :Gamm] = (Val[0:n_u - 2, :Gamm] + 10 * Val[1:n_u - 1, :Gamm] + Val[2:n_u,
                                                                                                     :Gamm]) / 12

            final_Thet_Val[n_u - 1, :Gamm] = (13 * Val[n_u - 1, :Gamm] + 15 * Val[n_u - 2, :Gamm]
                                              - 5 * Val[n_u - 3, :Gamm] + Val[n_u - 4, :Gamm]) / 48 \
                                             + (13 * Val[n_u - 1, 1:Gamm + 1] + 15 * Val[n_u, 1:Gamm + 1] - 5
                                                * Val[n_u + 1, 1:Gamm + 1] + Val[n_u + 2, 1:Gamm + 1]) / 48

            final_Thet_Val[n_u:K_grid - 1, :Gamm] = (Val[n_u - 1:K_grid - 2, 1:Gamm + 1] + 10
                                                     * Val[n_u:K_grid - 1, 1:Gamm + 1] + Val[n_u + 1:K_grid,
                                                                                         1:Gamm + 1]) / 12

            final_Thet_Val[K_grid - 1, :Gamm] = (13 * Val[K_grid - 1, 1:Gamm + 1] + 15 * Val[K_grid - 2, 1:Gamm + 1]
                                                 - 5 * Val[K_grid - 3, 1:Gamm + 1] + Val[K_grid - 4, 1:Gamm + 1]) / 48

            p[:] = torch.fft.ifft(toepM * torch.fft.fft(final_Thet_Val[:, :Gamm], dim=0), dim=0).real

            Val[:, :Gamm] = p[:K_grid, :]

            final_Thet_Val[0, Gamm] = (13 * Val[0, Gamm] + 15 * Val[1, Gamm]
                                       - 5 * Val[2, Gamm] + Val[3, Gamm]) / 48

            final_Thet_Val[1:n_u - 1, Gamm] = (Val[0:n_u - 2, Gamm] + 10 * Val[1:n_u - 1, Gamm]
                                               + Val[2:n_u, Gamm]) / 12

            final_Thet_Val[n_u - 1, Gamm] = (13 * Val[n_u - 1, Gamm] + 15 * Val[n_u - 2, Gamm]
                                             - 5 * Val[n_u - 3, Gamm] + Val[n_u - 4, Gamm]) / 48

            final_Thet_Val[n_u:K_grid, Gamm] = 0

            dim_reduction = (max(20, int(Gamm / 2)))

            p[:dim_reduction, :] = torch.fft.ifft(toepM[:dim_reduction]
                                                  * torch.fft.fft(final_Thet_Val[:dim_reduction, Gamm].unsqueeze(1),
                                                                  dim=0), dim=0).real

            Val[:dim_reduction, Gamm] = p[:dim_reduction, 0]

    @staticmethod
    def _interpolation(Val: torch.Tensor, nnot: int,  xmin: float, dx: float, interp_Atend: int, ):
        if interp_Atend != 1:
            price = Val[nnot - 1, 0]

        else:  # INTERPOLATION
            dd = 0 - (xmin + (nnot - 1) * dx)
            price = Val[nnot - 1, 0] + (Val[nnot, 0] - Val[nnot - 1, 0]) * dd / dx
        return max(0, price)

    @staticmethod
    def _price_computation(option_dict: Dict[str, Any], grid_dict: Dict[str, Any],
                           payoff_constants_dict: Dict[str, Any],stepSoftenerArray: torch.Tensor,
                           toepM: torch.Tensor, Thet: torch.Tensor,
                           Thet_vectorized: torch.Tensor,
                           p: torch.Tensor,final_Thet_Val: torch.Tensor, Val: torch.Tensor,):
        ProjStepPricer._Thet_initialization(Thet=Thet, S0=option_dict['S0'], K=option_dict['K'], xmin=grid_dict['xmin'],
                                            dx=grid_dict['dx'], a=grid_dict['a'],
                                            K_grid=grid_dict['K_grid'], varthet_01=payoff_constants_dict['varthet_01'],
                                            varthet_star=payoff_constants_dict['varthet_star'])

        ProjStepPricer._Val_initialization(stepSoftenerArray=stepSoftenerArray,  toepM=toepM, Thet=Thet,
                                           Thet_vectorized=Thet_vectorized, p=p,Val=Val, K_grid=grid_dict['K_grid'],
                                           n_u=grid_dict['n_h'], M=option_dict['M'])

        ProjStepPricer._Val_recursion(toepM=toepM[:, None],  p=p, final_Thet_Val=final_Thet_Val, Val=Val,
                                      M=option_dict['M'], K_grid=grid_dict['K_grid'], n_u=grid_dict['n_h'])

        Thet.zero_()

        return ProjStepPricer._interpolation(Val=Val, nnot=grid_dict['nnot'],
                                             xmin=grid_dict['xmin'],
                                             dx=grid_dict['dx'], interp_Atend=grid_dict['interp_Atend'])

    @staticmethod
    def _toepM_computation(beta: np.ndarray, K_grid: int,device: str):
        toepM = torch.from_numpy(np.concatenate([beta[K_grid - 1::-1], [0], beta[2 * K_grid - 2:K_grid - 1:-1]]))
        toepM = toepM.to(device=device)
        toepM = torch.fft.fft(toepM)
        return toepM

    @staticmethod
    def _up_and_out_call(option_dict: Dict[str, Any], grid_dict: Dict[str, Any], payoff_constants_dict: Dict[str, Any],
                         beta: np.ndarray, output: np.ndarray, cuda: bool):

        # Memory Pre-Allocation
        device = 'cpu'
        if cuda:
            device = 'cuda'

        stepSoftenerArray = ProjStepPricer._stepSoftener(stepRho=option_dict['stepRho'], M=option_dict['M'],
                                                         device=device)
        toepM = ProjStepPricer._toepM_computation(beta=beta, K_grid=grid_dict['K_grid'], device=device)
        # TODO : Try to use only Thet_vectorized and delete Thet
        Thet = torch.zeros(grid_dict['K_grid'], dtype=torch.float64, device=device)
        Thet_vectorized = torch.zeros((option_dict['M'] + 1, 2 * grid_dict['K_grid']), dtype=torch.float64,
                                      device=device)
        p = torch.zeros((2 * grid_dict['K_grid'], option_dict['M'] + 1), dtype=torch.float64, device=device)
        final_Thet_Val = torch.zeros((2 * grid_dict['K_grid'], option_dict['M'] + 2), dtype=torch.float64,
                                     device=device)
        Val = torch.zeros((grid_dict['K_grid'], option_dict['M'] + 2), dtype=torch.float64, device=device)



        K = option_dict['K']
        for index in prange(len(K)):
            option_dict['K'] = K[index]
            output[index] = ProjStepPricer._price_computation(option_dict=option_dict, grid_dict=grid_dict,
                                                              payoff_constants_dict=payoff_constants_dict,
                                                              stepSoftenerArray=stepSoftenerArray,
                                                              toepM=toepM,
                                                              Thet=Thet,
                                                              Thet_vectorized=Thet_vectorized,
                                                              p=p,final_Thet_Val=final_Thet_Val,
                                                              Val=Val,
                                                              )
