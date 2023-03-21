import numpy as np


def solve_dirichlet(a: np.array,
                    b: np.array,
                    c: np.array,
                    s: np.array,
                    u_left: float,
                    u_right: float) -> np.array:
    """
    Solve the tridiagonal set of equations for yvals:
        a[i] yvals[i + 1] + b[i] yvals[i] + c[i] yvals[i - 1] = s[i]
    for i = 1, ..., N - 2 (so len(yvals) == N). Note that these equations do not depend on the 0 or N - 1 elements of
    a, b, c, or s.
    Solves subject to the boundary condition yvals[0] = u_left, yvals[N-1] = u_right.

    """
    if not (len(a) == len(b) == len(c) == len(s)):
        raise ValueError(f"The lengths of arrays a, b, c, and s must be equal, "
                         f"they are {len(a)}, {len(b)}, {len(c)}, and {len(s)}")

    x = np.zeros(shape=len(a))
    y = np.zeros(shape=len(a))

    solution = np.zeros(shape=len(a))

    # Set left boundary conditions.
    x[1] = b[1]
    y[1] = s[1] - c[1] * u_left

    # Forwards pass
    for i in range(2, len(a)):
        x[i] = b[i] - c[i] * a[i - 1] / x[i - 1]
        y[i] = s[i] - c[i] * y[i - 1] / x[i - 1]

    # Set right boundary condition.
    solution[-1] = u_right
    solution[-2] = (y[-2] - a[-2] * solution[-1]) / x[-2]

    # Backwards pass
    for i in reversed(range(1, len(a) - 2)):
        solution[i] = (y[i] - a[i] * solution[i + 1]) / x[i]

    # Set left boundary condition.
    solution[0] = u_left

    return solution
