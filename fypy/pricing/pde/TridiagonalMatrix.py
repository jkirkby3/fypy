import numpy as np


class TridiagonalMatrix:
    def __init__(self, lower: np.array, diag: np.array, upper: np.array):
        if len(lower) != len(upper) != len(diag) - 1:
            raise ValueError("lengths of off-diagonals must be one less than the length of the diagonal")

        self._lower = lower
        self._diag = diag
        self._upper = upper

    def __mul__(self, vector: np.array) -> np.array:
        """
        d_0  u_0  0   ...  0    0           |    v_0      = 0               + v_0 d_0   + u_0 v_1
        l_0  d_1  u_1 ...  0    0           |    v_1      = l_0 v_0         + v_1 d_1   + u_1 v_2
        0    l_1  d_2 ...  0    0           |    v_2      = l_1 v_1         + v_2 d_2   + u_2 v_3
        ............. ...  ...  ...         |    ...      ...
                      ...  d_{N-1}  u_{N-1} |    v_{N-1}  =
                      ...  l_{N-1}  d_N     |    v_N      = l_{N-1} v_{N-1} + d_N v_N   + 0
        """

        # A_i = l_i * v_i
        A = self._lower * vector[:-1]
        # B_i = d_i * v_i
        B = self._diag * vector
        # C_i = u_i * v_{i + 1}
        C = self._upper * vector[1:]

        B[1:] += A
        B[:-1] += C
        return B
