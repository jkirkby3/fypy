from typing import Union, List
from scipy.interpolate import interp1d
import numpy as np


class LogLinearInterpolation(object):
    def __init__(self,
                 points: Union[np.ndarray, List],
                 values: Union[np.ndarray, List]):
        """
        Log linear interpolation. Performs linear interpolation of the logarithm of supplied values,
        and exponentiates. This is appropriate for discount curve interpolations
        :param points: Union[np.ndarray, List], the x-values
        :param values: Union[np.ndarray, List], the y-values
        """
        self._interp = interp1d(points, np.log(values), fill_value='extrapolate', bounds_error=False)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Main method to evaluate the interpolation
        :param x: Union[float, np.ndarray], the points at which to interpolate
        :return: Union[float, np.ndarray], the interpolated values, dimension matches the inputs
        """
        return np.exp(self._interp(x))
