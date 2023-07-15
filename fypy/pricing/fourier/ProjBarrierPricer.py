import numba
import numpy as np
from numba import prange

from fypy.model.levy.LevyModel import LevyModel
from fypy.pricing.fourier.ProjPricer import ProjPricer, LinearImpl


class ProjBarrierPricer(ProjPricer):
    def __init__(self, model: LevyModel, N: int = 2 ** 11, L: float = 10., order: int = 1,
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
        super().__init__(model, N, L, order, alpha_override)

    def price_strikes(self,
                      T: float = None,
                      M: float = None,
                      H: float = None,
                      down: bool = None,
                      rebate: float = None,
                      K: np.ndarray = None,
                      is_calls: np.ndarray = None) -> np.ndarray:
        """
        override of the method of StrikesPricer in order to handle new parameters

        """
        output = np.empty_like(K, dtype=float)
        self.price_strikes_fill(T=T, K=K, M=M, H=H, down=down,
                                rebate=rebate, is_calls=is_calls, output=output)
        return output

    def price(self,
              T: float = None,
              M: float = None,
              H: float = None,
              down: bool = None,
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
                           down: bool = None,
                           rebate: float = None,
                           K: np.ndarray = None,
                           is_calls: np.ndarray = None,
                           output: np.ndarray = None):

        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param M   = number of subintervals of [0,T] (total of M+1 monitoring points in time grid, including S_0)
        :param down = True for down and out (otherwise it's up and out)
        :param H    = barrier
        :param rebate = rebate paid immediately upon passing the barrier (knocking-out)
        :param is_calls: np.ndarray[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :return: None, this method fills in the output array, make sure its sized properly first
        # """

        # definition of option_params as a structured numpy array, in order to use numba
        # the array contains: S0, T, nrdt, nqdr, M, H, rebate

        dtype_option = [('S0', 'f8'),
                        ('T', 'f8'),
                        ('nrdt', 'f8'),
                        ('nqdt', 'f8'),
                        ('M', 'i4'),
                        ('H', 'f8'),
                        ('rebate', 'f8')
                        ]
        option_params = np.array([(self._model.spot(),
                                   T,
                                   self._get_dt_values(T=T, M=M)[0],
                                   self._get_dt_values(T=T, M=M)[1],
                                   int(M),
                                   H,
                                   rebate)],
                                 dtype=dtype_option)

        # definition of grid_params as a structured numpy array, in order to use numba
        # the array contains: dx, a, nnot, xmin, interp_Atend, grid_K,mult
        dtype_grid = [('dx', 'f8'),
                      ('a', 'f8'),
                      ('nnot', 'i4'),
                      ('xmin', 'float'),
                      ('interp_Atend', 'i4'),
                      ('grid_K', 'i4'),
                      ('mult', 'i4')]

        grid_params = np.array([(self.get_dx(T=T, K=K),
                                 1. / self.get_dx(T=T, K=K),
                                 0,
                                 0,
                                 0,
                                 int(self.get_N() / 2),
                                 1)],
                               dtype=dtype_grid)

        # DOWN AND OUT
        if down:
            # DOWN AND OUT CALL
            if is_calls[0]:
                self.down_and_out_call(option_params=option_params,
                                       grid_params=grid_params, K=K, is_calls=is_calls,
                                       output=output)

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
    def down_and_out_call(self, option_params: np.ndarray, grid_params: np.ndarray,
                          K: np.ndarray, is_calls: np.ndarray, output: np.ndarray):

        # update of numerical grid values, such as xmin, nnot, a
        ProjBarrierPricer._grid_update(option_params=option_params, grid_params=grid_params,
                                       is_calls=is_calls)

        # computation of basic integrals related to payoff coefficients
        varthet_01, varthet_star = ProjBarrierPricer.payoff_constants_computation_nb(dx=grid_params['dx'].item())

        # computation of nbar_array and impl_integrands
        nbar_array, impl_integrands = self._nbar_impl_computation(option_params=option_params,
                                                                  grid_params=grid_params, K=K)

        # update of the price using numba method
        price_doc_update_nb(option_params=option_params, grid_params=grid_params, N=self.get_N(),
                            nbar_array=nbar_array, impl_integrands=impl_integrands,
                            varthet_01=varthet_01, varthet_star=varthet_star,
                            K=K, output=output)

    def _get_dt_values(self, T: float, M: float):
        dt = T / M
        nrdt = -self.get_r(T) * dt
        nqdt = -self.get_q(T) * dt
        return nrdt, nqdt

    def get_dx(self, T: float, K: np.ndarray):
        lws_vec = np.log(K[0] / self.get_model().spot())

        cumulants = self.get_model().cumulants(T)
        alph = cumulants.get_truncation_heuristic(L=self.get_L()) \
            if np.isnan(self.get_alpha_override()) else self.get_alpha_override()

        # Ensure that grid is wide enough to cover the strike
        alph = max(alph, 1.15 * np.max(np.abs(lws_vec)) + cumulants.c1)
        return 2 * alph / (self.get_N() - 1)

    @staticmethod
    def _grid_update(option_params: np.ndarray, grid_params: np.ndarray, is_calls: np.ndarray):

        # The method _grid_update adjusts the parameters of the grid used
        # It modifies the grid based on the barrier and initial asset prices,
        # and handles special cases such as down-and-out call options
        # where the barrier price is near the initial asset price.
        # It also sets the grid spacing and the scaling factor for the grid

        l = np.log(option_params['H'].item() / option_params['S0'].item())
        grid_params['xmin'] = l
        grid_params['nnot'] = int(np.floor(1 - grid_params['xmin'].item() * grid_params['a'].item()))

        if grid_params['nnot'].item() >= grid_params['grid_K'].item():
            print(
                f"nnot is {grid_params['nnot'].item()} while grid_K is {grid_params['grid_K'].item()}, "
                f"need to increase alpha")

        if is_calls[0] == 1 and grid_params['nnot'].item() == 1:
            # In this case a DOC with H near S0 is still very valuable, so setting alph too small is bad idea
            grid_params['interp_Atend'] = 1  # Instruct to use interpolation at algorithm end
            # no change is made to dx
        else:
            nnot = max(2, int(np.floor(1 - grid_params['xmin'].item() * grid_params['a'].item())))
            grid_params['dx'] = l / (1 - nnot)

        grid_params['a'] = 1 / grid_params['dx'].item()

    @staticmethod
    def payoff_constants_computation_nb(dx: float):
        # computation of basic integrals related to payoff coefficients

        b3 = np.sqrt(15)
        b4 = np.sqrt(15) / 10
        varthet_01 = np.exp(.5 * dx) * (
                5 * np.cosh(b4 * dx) - b3 * np.sinh(
            b4 * dx) + 4) / 18
        varthet_m10 = np.exp(-.5 * dx) * (
                5 * np.cosh(b4 * dx) + b3 * np.sinh(
            b4 * dx) + 4) / 18
        varthet_star = varthet_01 + varthet_m10

        return varthet_01, varthet_star

    def _nbar_impl_computation(self, option_params: np.ndarray, grid_params: np.ndarray, K: np.ndarray):

        # TODO : try to set a common nbar for all the strikes, in order to extract more steps from the for cycle
        nbar_array = np.zeros(len(K), dtype=int)
        impl_integrands = np.zeros((len(K), self._N), dtype=complex)
        for index in range(len(K)):
            # nbar setting
            nbar_array[index] = self.get_nbar(a=grid_params['a'].item(),
                                              lws=np.log(K[index] / option_params['S0'].item()),
                                              lam=grid_params['xmin'].item())

            zmin = (1 - grid_params['grid_K'].item()) * grid_params['dx'].item()  # Kbar corresponds to zero

            impl = LinearImpl(N=self._N, dx=grid_params['dx'].item(), model=self._model,
                              T=option_params['T'].item() / option_params['M'].item(),
                              max_n_bar=nbar_array[index])

            impl_integrands[index, :] = impl.integrand(xmin=zmin)

        return nbar_array, impl_integrands

    @staticmethod
    def _beta_computation(option_params: np.ndarray = None, grid_params: np.ndarray = None,
                          integrand: np.ndarray = None, N: int = None):
        pass



# numba methods used inside price_doc_update_nb

@numba.jit(nopython=True, nogil=True)
def beta_computation_nb(mult: float, a: float, nrdt: float, N: int, integrand: np.ndarray):
    # computation of orthogonal projection coefficients
    Nmult = mult * N
    Cons = 24 * (a ** 2) * np.exp(nrdt) / Nmult

    return Cons * np.real(np.fft.fft(integrand))


@numba.jit(nopython=True, nogil=True)
def rho_zeta_computation_nb(S0: float, xmin: float, nbar: float, dx: float, a: float,
                            K: np.ndarray, index: int):
    # update of the difference (rho) and normalized difference (zeta) with respect to the nearest grid point
    rho = np.log(K[index] / S0) - (xmin + (nbar - 1) * dx)
    zeta = a * rho.item()

    return rho, zeta


@numba.jit(nopython=True, nogil=True)
def d_computation_nb(dx: float, zeta: float, rho: float):
    q_plus = (1 + np.sqrt(3 / 5)) / 2
    q_minus = (1 - np.sqrt(3 / 5)) / 2

    sigma = 1 - zeta
    sigma_plus = (q_plus - .5) * sigma
    sigma_minus = (q_minus - .5) * sigma

    es1 = np.exp(dx * sigma_plus)
    es2 = np.exp(dx * sigma_minus)

    dbar_0 = .5 + zeta * (.5 * zeta - 1)
    dbar_1 = sigma * (1 - .5 * sigma)

    d_0 = np.exp((rho + dx) * .5) * sigma ** 2 / 18 * (
            5 * ((1 - q_minus) * es2 + (1 - q_plus) * es1) + 4)
    d_1 = np.exp((rho - dx) * .5) * sigma / 18 * (
            5 * ((.5 * (zeta + 1) + sigma_minus) * es2 + (
            .5 * (zeta + 1) + sigma_plus) * es1) + 4 * (
                    zeta + 1))

    return d_0, dbar_0, d_1, dbar_1


@numba.jit(nopython=True, nogil=True)
def Thet_initial_update_nb(S0: float, rebate: float, xmin: float, nbar: float, dx: float, a: float, grid_K: int,
                           K: np.ndarray, varthet_01: float, varthet_star: float,
                           index: int):
    Thet = np.zeros(grid_K, dtype=float)

    # update of payoff integrals (Theta)

    # update of the difference (rho) and normalized difference (zeta) with respect to the nearest grid point
    rho, zeta = rho_zeta_computation_nb(S0=S0, xmin=xmin,
                                        nbar=nbar,
                                        dx=dx, a=a,
                                        K=K, index=index)

    d_0, dbar_0, d_1, dbar_1 = d_computation_nb(dx=dx, zeta=zeta, rho=rho)

    Thet[nbar - 1] = K[index] * (np.exp(-rho) * d_0 - dbar_0)
    Thet[nbar] = K[index] * (
            np.exp(dx - rho) * (varthet_01 + d_1) - (.5 + dbar_1))
    Thet[nbar + 1:grid_K] = \
        np.exp(xmin + dx * np.arange(nbar + 1, grid_K)) * S0 * varthet_star - K[index]

    Thet[0] = Thet[0] + 0.5 * rebate
    return Thet.flatten()


@numba.jit(nopython=True, nogil=True)
def val_rebate_computation_nb(rebate: float, grid_K: int,
                              beta: np.ndarray):
    if rebate != 0:
        val_rebate = np.zeros(grid_K)
        val_rebate[:grid_K - 1] = np.flip(np.cumsum(beta[:grid_K - 1]))
        val_rebate *= rebate
        return val_rebate  # NOTE: this includes the discounting via beta
    else:
        return np.zeros(grid_K)


@numba.jit(nopython=True, nogil=True)
def toepM_toepR_computation_nb(grid_K: int, beta: np.ndarray):
    toepM = np.empty(2 * grid_K)
    toepM[:grid_K] = beta[grid_K - 1::-1]
    toepM[grid_K] = 0.0
    toepM[grid_K + 1:] = beta[2 * grid_K - 2:grid_K - 1:-1]
    toepM = np.fft.fft(toepM)

    toepR = np.empty(2 * grid_K)
    toepR[:grid_K] = beta[2 * grid_K - 1:grid_K - 1:-1]
    toepR[grid_K] = 0.0
    toepR[grid_K + 1:] = np.zeros(grid_K - 1)
    toepR = np.fft.fft(toepR)

    return toepM, toepR


@numba.jit(nopython=True, nogil=True)
def Thetbars_computation_nb(S0: float, nrdt: float, dx: float, xmin: float, grid_K: int, K: np.ndarray,
                            beta: np.ndarray, varthet_star: float,
                            toepR: np.ndarray, index: int):
    Thetbar1 = np.exp(-nrdt) * K[index] * np.cumsum(
        beta[2 * grid_K - 1:grid_K - 1:-1])

    Thetbar2 = np.exp(-nrdt) * S0 * varthet_star * np.exp(
        xmin + dx * np.arange(grid_K, 2 * grid_K))

    Thetbar2_concat = np.zeros(len(toepR))
    Thetbar2_concat[:len(Thetbar2)] = Thetbar2

    Thetbar2_fft = np.fft.fft(Thetbar2_concat)

    p = np.real(np.fft.ifft(toepR * Thetbar2_fft))

    Thetbar2 = p[:grid_K]

    return Thetbar1, Thetbar2


@numba.jit(nopython=True, nogil=True)
def p_computation_nb(grid_K: int, Thet: np.ndarray, toepM: np.ndarray):
    len_Thet = grid_K
    len_zeros = len(toepM) - len_Thet
    Thet_extended = np.zeros(len_Thet + len_zeros)
    Thet_extended[:len_Thet] = Thet[:len_Thet]
    Thet_fft = np.fft.fft(Thet_extended)
    p = np.real(np.fft.ifft(toepM * Thet_fft))

    return p


@numba.jit(nopython=True, nogil=True)
def Thet_last_update_nb(Thet: np.ndarray, rebate: float, grid_K: int, Val: np.ndarray):
    Thet[1:grid_K - 1] = (Val[:grid_K - 2]
                          + 10 * Val[1:grid_K - 1] + Val[2:grid_K]) / 12

    Thet[0] = (13 * Val[0] + 15 * Val[1] - 5 * Val[2] + Val[3]) / 48
    Thet[grid_K - 1] = 2 * (13 * Val[grid_K - 1] + 15 * Val[grid_K - 2]
                            - 5 * Val[grid_K - 3] + Val[grid_K - 4]) / 48
    # NOTE: 2*theta(K) b/c of augmenting

    Thet[0] = Thet[0] + 0.5 * rebate  # account for overhang into the knock-out region

    return Thet


@numba.jit(nopython=True, nogil=True)
def Val_computation_nb(M: int, rebate: float, nrdt: float, nqdt: float, grid_K: int,
                       Thet: np.ndarray, val_rebate: np.ndarray, toepM: np.ndarray,
                       Thetbar1: np.ndarray, Thetbar2: np.ndarray,
                       p: np.ndarray):
    if rebate != 0:
        Val = p[:grid_K] + np.exp(nrdt) * (
                Thetbar2 - Thetbar1) + val_rebate
    else:
        Val = p[:grid_K] + np.exp(nrdt) * (Thetbar2 - Thetbar1)

    for m in range(M - 2, -1, -1):
        Thet = Thet_last_update_nb(Thet=Thet, rebate=rebate, grid_K=grid_K, Val=Val)

        Thet_extended = np.zeros(2 * grid_K)

        Thet_extended[:grid_K] = Thet[:grid_K]

        p = np.fft.ifft(toepM * np.fft.fft(Thet_extended))
        Val[:grid_K] = np.real(p[:grid_K] + np.exp(
            nqdt * (M - m - 1)) * Thetbar2 - np.exp(
            nrdt * (M - m - 1)) * Thetbar1)

        if rebate != 0:
            Val = Val + val_rebate
    return Val


@numba.jit(nopython=True, nogil=True)
def price_final_adjustment_nb(interp_Atend: int, xmin: float, nnot: float, dx: float,
                              Val: np.ndarray, index: int, output: np.ndarray):
    if interp_Atend == 1:
        dd = 0 - (xmin + (nnot - 1) * dx)
        price = Val[nnot - 1].item() + (
                Val[nnot].item() - Val[
            nnot - 1].item()) * dd / dx  # ie linear interp of nnot and nnot+1
        output[index] = max(0.0, price)


    else:
        price = Val[nnot - 1].item()
        output[index] = max(0.0, price)


# TODO : Understand how to fully take advantage of Numba's improvements
@numba.jit(nopython=True, nogil=True, parallel=True)
def price_doc_update_nb(option_params: np.ndarray, grid_params: np.ndarray,
                        nbar_array: np.ndarray, impl_integrands: np.ndarray,
                        varthet_01: float, varthet_star: float,
                        N: int, K: np.ndarray, output: np.ndarray):

    for index in prange(len(K)):
        beta = beta_computation_nb(mult=grid_params['mult'].item(), a=grid_params['a'].item(),
                                   nrdt=option_params['nrdt'].item(), N=N,
                                   integrand=impl_integrands[index])

        # update of payoff integrals (Theta)
        Thet = Thet_initial_update_nb(S0=option_params['S0'].item(), rebate=option_params['rebate'].item(),
                                      xmin=grid_params['xmin'].item(), nbar=nbar_array[index].item(),
                                      dx=grid_params['dx'].item(), a=grid_params['a'].item(),
                                      grid_K=grid_params['grid_K'].item(), K=K,
                                      varthet_01=varthet_01, varthet_star=varthet_star, index=index)

        # computation of the parameter val_rebate, depending on the option parameter "rebate"
        val_rebate = val_rebate_computation_nb(rebate=option_params['rebate'].item(),
                                               grid_K=grid_params['grid_K'].item(), beta=beta)

        toepM, toepR = toepM_toepR_computation_nb(grid_K=grid_params['grid_K'].item(), beta=beta)

        Thetbar1, Thetbar2 = Thetbars_computation_nb(S0=option_params['S0'].item(),
                                                     nrdt=option_params['nrdt'].item(),
                                                     dx=grid_params['dx'].item(), xmin=grid_params['xmin'].item(),
                                                     grid_K=grid_params['grid_K'].item(), K=K,
                                                     beta=beta,
                                                     varthet_star=varthet_star,
                                                     toepR=toepR, index=index)

        p = p_computation_nb(grid_K=grid_params['grid_K'].item(), Thet=Thet, toepM=toepM)

        Val = Val_computation_nb(M=option_params['M'].item(), rebate=option_params['rebate'].item(),
                                 nrdt=option_params['nrdt'].item(), nqdt=option_params['nqdt'].item(),
                                 grid_K=grid_params['grid_K'].item(),
                                 Thet=Thet, val_rebate=val_rebate, toepM=toepM,
                                 Thetbar1=Thetbar1, Thetbar2=Thetbar2,
                                 p=p)

        price_final_adjustment_nb(interp_Atend=grid_params['interp_Atend'].item(),
                                  xmin=grid_params['xmin'].item(),
                                  nnot=grid_params['nnot'].item(),
                                  dx=grid_params['dx'].item(),
                                  Val=Val, index=index, output=output)
