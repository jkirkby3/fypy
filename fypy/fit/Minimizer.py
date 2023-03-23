from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Tuple, Callable, Optional
from scipy.optimize import least_squares, minimize


class OptResult(object):
    def __init__(self,
                 params: np.ndarray,
                 value: float,
                 success: bool,
                 message: str = ""):
        """
        Result of optimization over parameters.
        :param params: np.ndarray, the optimal parameters
        :param value: float, the optimal objective value
        :param success: bool, true if optimization succeeded, else false
        :param message: str, messaging indicating status of optimization
        """
        self.params = params
        self.value = value
        self.success = success
        self.message = message


class Minimizer(ABC):
    """
    Abstract base class for a minimizer. Calibration routines will take an instance
    of the abstract class, allowing you to swap in your favorite minimizer, or use
    the one provided by the framework
    """
    @abstractmethod
    def minimize(self,
                 function: Callable,
                 bounds: Union[Tuple, List[Tuple]] = None,
                 guess: np.ndarray = None,
                 constraints=()) -> OptResult:
        """
        Minimize the objectives to obtain optimal params. Main function to override
        :param function: Callable, the objective function to minimizer
        :param bounds: list of Tuple, each tuple is a (lower, upper) bound pair for a param
            Use np.inf with an appropriate sign to disable bounds on all or some variables.
        :param guess: np.ndarray, initial guess of parameters
        :return: OptResult, the result of optimization
        """
        raise NotImplementedError


class LeastSquares(Minimizer):
    def __init__(self,
                 method: str = "trf",
                 max_nfev: int = None,
                 ftol: float = 1e-08,
                 xtol: float = 1e-08,
                 gtol: float = 1e-08,
                 x_scale: float = 1.,
                 loss: str = 'linear',
                 verbose: int = 0):
        """
        Least squares minimizer. This is a light wrapper on top of scipy.least_squares.

        For more details, see scipy documentation:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        :param method: str, the scipy least_squares optimization method, 'trf’, ‘dogbox’, ‘lm’.
            Note that LevMar (lm) doesnt support param bounds
        :param max_nfev: Maximum number of function evaluations before the termination.
            If None (default), the value is chosen automatically (according to method)
        :param ftol: Tolerance for termination by the change of the cost function. Default is 1e-8.
            The optimization process is stopped when dF < ftol * F, and there was an adequate agreement between a local
            quadratic model and the true model in the last step. If None and ‘method’ is not ‘lm’,
            the termination by this condition is disabled. If ‘method’ is ‘lm’, this tolerance must be higher
            than machine epsilon.
        :param xtol: Tolerance for termination by the change of the independent variables. Default is 1e-8.
            The exact condition depends on the method used:
        :param gtol: Tolerance for termination by the norm of the gradient. Default is 1e-8.
            The exact condition depends on a method used
        :param x_scale: Characteristic scale of each variable. Setting x_scale is equivalent to reformulating the
            problem in scaled variables xs = x / x_scale. An alternative view is that the size of a trust region along
            jth dimension is proportional to x_scale[j]. Improved convergence may be achieved by setting x_scale such
            that a step of a given size along any of the scaled variables has a similar effect on the cost function.
            If set to ‘jac’, the scale is iteratively updated using the inverse norms of the columns of the Jacobian
            matrix
        :param loss: Determines the loss function. The following keyword values are allowed:
            ‘linear’ (default) : rho(z) = z. Gives a standard least-squares problem.
            ‘soft_l1’ : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss.
                Usually a good choice for robust least squares.
            ‘huber’ : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similarly to ‘soft_l1’.
        :param verbose: Level of algorithm’s verbosity:
            0 (default) : work silently.
            1 : display a termination report.
            2 : display progress during iterations (not supported by ‘lm’ method).
        """
        self._method = method
        self._max_iter = max_nfev
        self._ftol = ftol
        self._xtol = xtol
        self._gtol = gtol
        self._x_scale = x_scale
        self._loss = loss
        self._verbose = verbose

    def minimize(self,
                 function: Callable,
                 bounds: Union[Tuple, List[Tuple]] = None,
                 guess: np.ndarray = None,
                 constraints=()) -> OptResult:
        """
        Minimize the objectives to obtain optimal params.
        :param function: Callable, the objective function to minimizer
        :param bounds: list of Tuple, each tuple is a (lower, upper) bound pair for a param
            Use np.inf with an appropriate sign to disable bounds on all or some variables.
        :param guess: np.ndarray, initial guess of parameters
        :return: OptResult, the result of optimization
        """
        if constraints:
            raise NotImplementedError("Least Squares currently doesnt support constraints")

        if isinstance(bounds, List):
            # Convert into least_squares library convention
            bounds = ([b[0] for b in bounds], [b[1] for b in bounds])

        fit = least_squares(function,
                            x0=guess,
                            bounds=bounds,
                            max_nfev=self._max_iter,
                            ftol=self._ftol,
                            xtol=self._xtol,
                            x_scale=self._x_scale,
                            method=self._method,
                            loss=self._loss,
                            verbose=self._verbose)

        return OptResult(params=fit.x, value=fit.fun, success=fit.success,
                         message=fit.message)


class ScipyMinimizer(Minimizer):
    def __init__(self,
                 method: str,
                 tol: Optional[float] = None,
                 options: dict = None):
        self._method = method
        self._tol = tol
        self._options = options

    def minimize(self,
                 function: Callable,
                 bounds: Union[Tuple, List[Tuple]] = None,
                 guess: np.ndarray = None,
                 constraints=()) -> OptResult:
        """"""
        fit = minimize(fun=function, x0=guess, method=self._method,
                       bounds=bounds, constraints=constraints,
                       options=self._options, tol=self._tol)

        return OptResult(params=fit.x, value=fit.fun, success=fit.success, message=fit.message)
