import math


def binomial_lattice_black_scholes(S_0: float,
                                   K: float,
                                   r: float,
                                   T: float,
                                   sigma: float,
                                   M: int,
                                   call: int,
                                   is_american: bool) -> float:
    """
    Calculate price option using Binomial Lattice
    :param K: float, Option Strike
    :param S_0: float, Initial Price of asset
    :param r: float, Interest rate
    :param T: float, Time to maturity
    :param sigma: float, Vol of asset
    :param M: int, Number of steps
    :param call: int, Call or put
    :param is_american: bool, Type of option
    """

    if call == 1:
        def payoff(C, S):
            return max(C, S - K)
    else:
        def payoff(C, S):
            return max(C, K - S)

    delta_t = T / M
    up_move = math.exp(sigma * math.sqrt(delta_t))
    down_move = 1 / up_move
    p = (math.exp(r * delta_t) - down_move) / (up_move - down_move)
    discount = math.exp(-r * delta_t)
    prob_up = discount * p
    prob_down = discount * (1 - p)
    stock_matrix = [0] * (2 * M + 1)
    stock_matrix[M] = S_0
    for i in range(1, M + 1):
        stock_matrix[M + i] = up_move * stock_matrix[M + i - 1]
        stock_matrix[M - i] = down_move * stock_matrix[M - i + 1]

    # Terminal Payoff
    Pvals = [0] * (2 * M + 1)
    for i in range(0, 2 * M + 1, 2):
        Pvals[i] = payoff(0, stock_matrix[i])

    # Calculate Price Recursively
    if is_american:
        for iter in range(M):
            for i in range(iter + 1, 2 * M + 1 - iter, 2):
                cont = prob_up * Pvals[i + 1] + prob_down * Pvals[i - 1]
                Pvals[i] = payoff(cont, stock_matrix[i])
    else:
        for iter in range(M):
            for i in range(iter + 1, 2 * M + 1 - iter, 2):
                Pvals[i] = prob_up * Pvals[i + 1] + prob_down * Pvals[i - 1]

    price = Pvals[M]
    return price
