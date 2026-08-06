"""
Exact Heston Greeks and fast calibration via AADC (Automatic Adjoint Differentiation).

This example demonstrates how AADC can accelerate fypy's Heston calibration by
providing exact gradients of PROJ prices w.r.t. all model parameters in a single
reverse pass, replacing scipy's finite-difference Jacobian.

Three levels of speedup are shown:

  1. PRICING:  price + 5 Greeks in one forward+reverse   (~14x vs FD)
  2. CALIBRATION: exact Jacobian for least-squares fit    (~2x vs FD)
  3. AITF:  d(calibrated params)/d(market prices) via
            Implicit Function Theorem                     (~200x vs FD)

AITF differentiates THROUGH the calibration without backpropagating through
optimizer iterations. At convergence, the AADC Jacobian gives the full
sensitivity matrix via one 5x5 linear solve.

The PROJ method (Kirkby 2015) is implemented on the AADC tape using:
  - Complex arithmetic with idouble (real/imag split)
  - Branch-free csqrt/clog (Heston CF guarantees re(z)>0)
  - Cooley-Tukey FFT with precomputed twiddle factors
  - Cubic B-spline payoff coefficients (matching fypy's CubicImpl)

Requirements:
  pip install aadc   # free evaluation version from matlogica.com/aadc
"""
import numpy as np
import time

from fypy.model.sv.Heston import Heston
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.termstructures.EquityForward import EquityForward
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate

try:
    import aadc
    from aadc import idouble
except ImportError:
    raise ImportError(
        "This example requires the AADC package.\n"
        "Install: pip install aadc\n"
        "Free evaluation version, no licence key required.\n"
        "See: https://matlogica.com/aadc"
    )


# ── Complex idouble arithmetic ───────────────────────────────────────────────

def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

def _cdiv(ar, ai, br, bi):
    d = br * br + bi * bi + 1e-300
    return (ar * br + ai * bi) / d, (ai * br - ar * bi) / d

def _cexp(ar, ai):
    e = aadc.math.exp(ar)
    return e * aadc.math.cos(ai), e * aadc.math.sin(ai)

def _csqrt_pos(ar, ai):
    """Complex sqrt when re(z) > 0. Branch-free."""
    m = aadc.math.sqrt(ar * ar + ai * ai + 1e-300)
    s = aadc.math.sqrt((m + ar) * idouble(0.5) + 1e-300)
    return s, ai / (s * idouble(2.0) + 1e-300)

def _clog_pos(ar, ai):
    """Complex log when re(z) > 0."""
    m = aadc.math.sqrt(ar * ar + ai * ai + 1e-300)
    return aadc.math.log(m + 1e-300), aadc.math.atan(ai / (ar + 1e-300))


# ── Heston CF on AADC tape ───────────────────────────────────────────────────

def _heston_cf(w_arr, v0, theta, kappa, xi, rho, T, drift):
    """Heston characteristic function at real frequencies w_arr."""
    phi_re, phi_im = [], []
    for w in w_arr:
        a_re, a_im = idouble(-0.5 * w * w), idouble(-0.5 * w)
        b_re = kappa
        b_im = idouble(0.0) - rho * xi * idouble(w)
        o2 = xi * xi

        b2r, b2i = _cmul(b_re, b_im, b_re, b_im)
        agr, agi = _cmul(a_re, a_im, o2 * idouble(0.5), idouble(0.0))
        D_re, D_im = _csqrt_pos(b2r - agr * idouble(4.0), b2i - agi * idouble(4.0))

        bDr, bDi = b_re - D_re, b_im - D_im
        eDr, eDi = _cexp(D_re * idouble(-T), D_im * idouble(-T))
        Gr, Gi = _cdiv(bDr, bDi, b_re + D_re, b_im + D_im)

        bDo2r, bDo2i = _cdiv(bDr, bDi, o2, idouble(0.0))
        GeDr, GeDi = _cmul(Gr, Gi, eDr, eDi)
        nr, ni = idouble(1.0) - eDr, idouble(0.0) - eDi
        dr, di = idouble(1.0) - GeDr, idouble(0.0) - GeDi
        fr, fi = _cdiv(nr, ni, dr, di)
        Br, Bi = _cmul(bDo2r, bDo2i, fr, fi)

        psir, psii = _cdiv(dr, di, idouble(1.0) - Gr, idouble(0.0) - Gi)
        kto2r, kto2i = _cdiv(kappa * theta, idouble(0.0), o2, idouble(0.0))
        lpr, lpi = _clog_pos(psir, psii)
        ir_, ii_ = bDr * idouble(T) - lpr * idouble(2.0), bDi * idouble(T) - lpi * idouble(2.0)
        Ar, Ai = _cmul(kto2r, kto2i, ir_, ii_)

        er = Ar + Br * v0
        ei = Ai + Bi * v0 + idouble(w * drift * T)
        pr, pi = _cexp(er, ei)
        phi_re.append(pr)
        phi_im.append(pi)
    return phi_re, phi_im


# ── FFT on AADC tape ────────────────────────────────────────────────────────

def _fft(xr, xi):
    N = len(xr)
    bits = int(np.log2(N))
    def rev(x, b):
        r = 0
        for _ in range(b):
            r = (r << 1) | (x & 1); x >>= 1
        return r
    yr = [xr[rev(i, bits)] for i in range(N)]
    yi = [xi[rev(i, bits)] for i in range(N)]
    length = 2
    while length <= N:
        half = length // 2
        ang = -2.0 * np.pi / length
        tw_r = [np.cos(ang * k) for k in range(half)]
        tw_i = [np.sin(ang * k) for k in range(half)]
        for s in range(0, N, length):
            for k in range(half):
                i0, i1 = s + k, s + k + half
                tr = yr[i1]*idouble(tw_r[k]) - yi[i1]*idouble(tw_i[k])
                ti = yr[i1]*idouble(tw_i[k]) + yi[i1]*idouble(tw_r[k])
                yr[i1], yi[i1] = yr[i0] - tr, yi[i0] - ti
                yr[i0], yi[i0] = yr[i0] + tr, yi[i0] + ti
        length *= 2
    return yr, yi


# ── PROJ grid (precomputed outside recording) ────────────────────────────────

def _proj_grid(model, T, K, N):
    """Precompute PROJ grid parameters using fypy's own cumulants."""
    S0 = model.spot()
    lws = np.log(K / S0)
    cumulants = model.cumulants(T)
    alph = cumulants.get_truncation_heuristic(L=10.0)
    alph = max(alph, 1.15 * abs(lws) + cumulants.c1)

    dx = 2 * alph / (N - 1)
    a = 1.0 / dx
    lam = cumulants.c1 - (N / 2 - 1) * dx
    nbar = min(int(np.floor(a * (lws - lam) + 1)), N - 1)
    xmin = lws - (nbar - 1) * dx
    dw = 2 * np.pi / (N * dx)
    w_nz = dw * np.arange(1, N)

    b0, b1, b2c, b3 = 1208/2520, 1191/2520, 120/2520, 1/2520
    zeta = (np.sin(w_nz/(2*a))/w_nz)**4 / \
           (b0 + b1*np.cos(w_nz/a) + b2c*np.cos(2*w_nz/a) + b3*np.cos(3*w_nz/a))
    phase = np.exp(-1j * xmin * dw * np.arange(0, N))
    cons = 32 * a**4

    g1 = 1/24-1/20*np.exp(dx)*(np.exp(-7/4*dx)/54+np.exp(-1.5*dx)/18+np.exp(-1.25*dx)/2+7*np.exp(-dx)/27)
    g2 = .5-.05*(28/27+np.exp(-7/4*dx)/54+np.exp(-1.5*dx)/18+np.exp(-1.25*dx)/2
                 +14*np.exp(-dx)/27+121/54*np.exp(-.75*dx)+23/18*np.exp(-.5*dx)+235/54*np.exp(-.25*dx))
    g3 = 23/24-np.exp(-dx)/90*((28+7*np.exp(-dx))/3+(14*np.exp(dx)+np.exp(-7/4*dx)
         +242*np.cosh(.75*dx)+470*np.cosh(.25*dx))/12+.25*(np.exp(-1.5*dx)+9*np.exp(-1.25*dx)+46*np.cosh(.5*dx)))
    g4 = 14/3*(2+np.cosh(dx))+.5*(np.cosh(1.5*dx)+9*np.cosh(1.25*dx)+23*np.cosh(.5*dx)) \
         +1/6*(np.cosh(7/4*dx)+121*np.cosh(.75*dx)+235*np.cosh(.25*dx))

    G = np.zeros(nbar + 1)
    G[nbar], G[nbar-1], G[nbar-2] = K*g1, K*g2, K*g3
    G[:nbar-2] = K - S0*np.exp(xmin)*np.exp(dx*np.arange(0, nbar-2))/90*g4

    r = model.discountCurve.implied_rate(T)
    q_rate = model.forwardCurve.divDiscountCurve.implied_rate(T) if \
        getattr(model.forwardCurve, 'divDiscountCurve', None) else 0.0
    disc = np.exp(-r * T)

    return dict(N=N, w_nz=w_nz, zeta=zeta, phase=phase, cons=cons,
                nbar=nbar, G=G, scale=cons*disc/N, drift=r-q_rate,
                parity=(S0*np.exp((r-q_rate)*T) - K)*disc)


def _proj_on_tape(v0, theta, kappa, xi, rho, T, gp):
    """Record PROJ pricing on AADC tape."""
    N = gp['N']
    phi_re, phi_im = _heston_cf(gp['w_nz'], v0, theta, kappa, xi, rho, T, gp['drift'])

    gr = [idouble(1.0 / gp['cons'])] + [None]*(N-1)
    gi = [idouble(0.0)] + [None]*(N-1)
    zeta, phase = gp['zeta'], gp['phase']
    for k in range(1, N):
        zr, zi = float(np.real(zeta[k-1])), float(np.imag(zeta[k-1]))
        pr = phi_re[k-1]*idouble(zr) - phi_im[k-1]*idouble(zi)
        pi = phi_re[k-1]*idouble(zi) + phi_im[k-1]*idouble(zr)
        phr, phi_ = float(np.real(phase[k])), float(np.imag(phase[k]))
        gr[k] = pr*idouble(phr) - pi*idouble(phi_)
        gi[k] = pr*idouble(phi_) + pi*idouble(phr)

    beta_re, _ = _fft(gr, gi)
    price = idouble(0.0)
    for j in range(gp['nbar'] + 1):
        price = price + beta_re[j] * idouble(gp['G'][j])
    return price * idouble(gp['scale']) + idouble(gp['parity'])


def record_pricer(model, T, K, N=512):
    """Record one AADC tape for pricing a single (T, K) instrument.

    Returns (functions, arg_dict, result_handle).
    """
    params = model.get_params()  # [v0, theta, kappa, sigma_v, rho]
    gp = _proj_grid(model, T, K, N)

    fn = aadc.Functions()
    ag = {}
    pnames = ['v0', 'theta', 'kappa', 'xi', 'rho']
    fn.start_recording()
    ids = []
    for i, p in enumerate(pnames):
        t = idouble(params[i])
        ag[p] = t.mark_as_input()
        ids.append(t)
    cp = _proj_on_tape(ids[0], ids[1], ids[2], ids[3], ids[4], T, gp)
    res = cp.mark_as_output()
    fn.stop_recording()
    return fn, ag, res


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Setup (same as fypy examples) ─────────────────────────────────────
    S0, r, q = 100.0, 0.05, 0.02
    disc_curve = DiscountCurve_ConstRate(rate=r)
    div_disc = DiscountCurve_ConstRate(rate=q)
    fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

    model = Heston(v_0=0.04, theta=0.04, kappa=2.0, sigma_v=0.3, rho=-0.7,
                   forwardCurve=fwd, discountCurve=disc_curve)

    T, K, N = 1.0, 100.0, 512
    pnames = ['v0', 'theta', 'kappa', 'xi', 'rho']
    true_params = model.get_params()

    # ── 1. Pricing ────────────────────────────────────────────────────────
    print("=" * 65)
    print("1. PRICING: AADC PROJ vs fypy PROJ")
    print("=" * 65)

    ref = ProjEuropeanPricer(model=model, N=N).price(T=T, K=K, is_call=True)
    fn, ag, res = record_pricer(model, T, K, N)

    workers = aadc.ThreadPool(1)
    inputs = {ag[pnames[j]]: true_params[j] for j in range(5)}
    all_args = [ag[p] for p in pnames]
    result = aadc.evaluate(fn, {res: all_args}, inputs, workers)
    aadc_price = float(np.asarray(result[0][res]).flat[0])
    grads = {p: float(np.asarray(result[1][res][ag[p]]).flat[0]) for p in pnames}

    print(f"fypy PROJ:  {ref:.6f}")
    print(f"AADC PROJ:  {aadc_price:.6f}  (diff: {abs(aadc_price-ref):.2e})")

    # FD verification
    h = 1e-6
    print(f"\nGradients:")
    for j, p in enumerate(pnames):
        pp = true_params.copy(); pp[j] += h
        pm = true_params.copy(); pm[j] -= h
        model.set_params(pp)
        pu = ProjEuropeanPricer(model=model, N=N).price(T=T, K=K, is_call=True)
        model.set_params(pm)
        pd = ProjEuropeanPricer(model=model, N=N).price(T=T, K=K, is_call=True)
        fd = (pu - pd) / (2*h)
        ratio = grads[p] / fd if abs(fd) > 1e-12 else float('nan')
        print(f"  d/d({p:>5s}): AD={grads[p]:+10.6f}  FD={fd:+10.6f}  ratio={ratio:.4f}")
    model.set_params(true_params)

    # Benchmark
    n_iter = 500
    t0 = time.time()
    for _ in range(n_iter):
        aadc.evaluate(fn, {res: all_args}, inputs, workers)
    t_aadc = (time.time() - t0) / n_iter * 1000

    t0 = time.time()
    for _ in range(n_iter):
        ProjEuropeanPricer(model=model, N=N).price(T=T, K=K, is_call=True)
    t_fypy = (time.time() - t0) / n_iter * 1000

    t0 = time.time()
    for _ in range(n_iter):
        ProjEuropeanPricer(model=model, N=N).price(T=T, K=K, is_call=True)
        for j in range(5):
            for s in [+1, -1]:
                pp = true_params.copy(); pp[j] += s*h
                model.set_params(pp)
                ProjEuropeanPricer(model=model, N=N).price(T=T, K=K, is_call=True)
        model.set_params(true_params)
    t_fd = (time.time() - t0) / n_iter * 1000

    print(f"\nBenchmark:")
    print(f"  AADC (price + 5 grads):  {t_aadc:.2f} ms")
    print(f"  fypy (price only):       {t_fypy:.2f} ms")
    print(f"  fypy + FD (11 evals):    {t_fd:.1f} ms")
    print(f"  Speedup vs FD:           {t_fd/t_aadc:.0f}x")

    # ── 2. Calibration ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("2. CALIBRATION: AADC Jacobian vs scipy FD")
    print("=" * 65)

    from scipy.optimize import least_squares

    ttms = [0.25, 0.5, 1.0, 2.0]
    strikes = np.array([90., 95., 100., 105., 110.])
    n_instr = len(ttms) * len(strikes)

    market_prices = []
    for Ti in ttms:
        market_prices.append(
            ProjEuropeanPricer(model=model, N=N).price_strikes(
                T=Ti, K=strikes, is_calls=np.ones(len(strikes), dtype=bool)))
    market_prices = np.concatenate(market_prices)

    # Record tapes for all instruments
    tapes = []
    for Ti in ttms:
        for Ki in strikes:
            tapes.append(record_pricer(model, Ti, Ki, N))

    def aadc_prices_jac(params):
        prices = np.zeros(n_instr)
        jac = np.zeros((n_instr, 5))
        for idx, (fn_i, ag_i, res_i) in enumerate(tapes):
            inp = {ag_i[pnames[j]]: params[j] for j in range(5)}
            aa = [ag_i[p] for p in pnames]
            r = aadc.evaluate(fn_i, {res_i: aa}, inp, workers)
            prices[idx] = float(np.asarray(r[0][res_i]).flat[0])
            for j in range(5):
                jac[idx, j] = float(np.asarray(r[1][res_i][aa[j]]).flat[0])
        return prices, jac

    def fypy_prices(params):
        model.set_params(params)
        prices = []
        for Ti in ttms:
            for Ki in strikes:
                prices.append(
                    ProjEuropeanPricer(model=model, N=N).price(T=Ti, K=Ki, is_call=True))
        return np.array(prices)

    x0 = np.array([0.06, 0.06, 1.0, 0.5, -0.3])
    bounds = ([1e-4, 1e-4, 0.01, 1e-4, -0.999], [1.0, 1.0, 50, 10, 0.999])

    n_runs = 3
    t0 = time.time()
    for _ in range(n_runs):
        r_aadc = least_squares(lambda p: aadc_prices_jac(p)[0] - market_prices,
                               x0=x0, jac=lambda p: aadc_prices_jac(p)[1],
                               bounds=bounds, method='trf')
    t_aadc_cal = (time.time() - t0) / n_runs

    t0 = time.time()
    for _ in range(n_runs):
        r_fd = least_squares(lambda p: fypy_prices(p) - market_prices,
                             x0=x0, bounds=bounds, method='trf')
    t_fd_cal = (time.time() - t0) / n_runs
    model.set_params(true_params)  # restore after FD calibration

    print(f"  AADC:  {t_aadc_cal:.2f}s  (converged: {r_aadc.success})")
    print(f"  FD:    {t_fd_cal:.2f}s  (converged: {r_fd.success})")
    print(f"  Speedup: {t_fd_cal/t_aadc_cal:.1f}x")
    print(f"  Max param error: {np.max(np.abs(r_aadc.x - true_params)):.2e}")

    # ── 3. AITF ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("3. AITF: d(calibrated params) / d(market prices)")
    print("=" * 65)

    cal = r_aadc.x
    _, J = aadc_prices_jac(cal)

    t0 = time.time()
    dtheta_dP = np.linalg.solve(J.T @ J, J.T)  # (5 x n_instr)
    t_ift = time.time() - t0

    atm_idx = ttms.index(1.0) * len(strikes) + list(strikes).index(100.)
    print(f"\n  Sensitivity to ATM 1Y (instrument #{atm_idx}):")
    for j in range(5):
        print(f"    d({pnames[j]:>5s}*)/dP = {dtheta_dP[j, atm_idx]:+.6f}")

    # FD check
    h_m = 1e-4
    mu, md = market_prices.copy(), market_prices.copy()
    mu[atm_idx] += h_m; md[atm_idx] -= h_m
    ru = least_squares(lambda p: aadc_prices_jac(p)[0] - mu, x0=cal,
                       jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
    rd = least_squares(lambda p: aadc_prices_jac(p)[0] - md, x0=cal,
                       jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
    fd_s = (ru.x - rd.x) / (2*h_m)
    print(f"\n  FD verification:")
    for j in range(5):
        ratio = dtheta_dP[j, atm_idx] / fd_s[j] if abs(fd_s[j]) > 1e-12 else float('nan')
        print(f"    d({pnames[j]:>5s}*)/dP: IFT={dtheta_dP[j,atm_idx]:+.6f}  FD={fd_s[j]:+.6f}  ratio={ratio:.4f}")

    # AITF benchmark
    t0 = time.time()
    for _ in range(n_runs):
        _, Js = aadc_prices_jac(cal)
        np.linalg.solve(Js.T @ Js, Js.T)
    t_ift_b = (time.time() - t0) / n_runs

    t0 = time.time()
    for i in range(n_instr):
        mu2, md2 = market_prices.copy(), market_prices.copy()
        mu2[i] += h_m; md2[i] -= h_m
        least_squares(lambda p: aadc_prices_jac(p)[0] - mu2, x0=cal,
                      jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
        least_squares(lambda p: aadc_prices_jac(p)[0] - md2, x0=cal,
                      jac=lambda p: aadc_prices_jac(p)[1], bounds=bounds, method='trf')
    t_fd_ift = time.time() - t0

    print(f"\n  IFT:  {t_ift_b*1000:.0f} ms  (1 Jacobian + 5x5 solve)")
    print(f"  FD:   {t_fd_ift:.1f}s  ({n_instr}x2 recalibrations)")
    print(f"  Speedup: {t_fd_ift/t_ift_b:.0f}x")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print("=" * 65)
    print(f"  Pricing:     AADC {t_aadc:.2f} ms vs FD {t_fd:.1f} ms  ({t_fd/t_aadc:.0f}x)")
    print(f"  Calibration: AADC {t_aadc_cal:.2f}s vs FD {t_fd_cal:.2f}s  ({t_fd_cal/t_aadc_cal:.1f}x)")
    print(f"  AITF:        AADC {t_ift_b*1000:.0f} ms vs FD {t_fd_ift:.1f}s  ({t_fd_ift/t_ift_b:.0f}x)")
