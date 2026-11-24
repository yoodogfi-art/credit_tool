"""
Term-structure models -> spot (zero) curve -> forward rates.

Four models, numpy only:
  * Nelson-Siegel (NS)            : 4 params, tau grid-searched, betas by OLS
  * Nelson-Siegel-Svensson (NSS)  : 6 params, (tau1, tau2) grid, betas by OLS
  * Cubic Spline                  : natural cubic regression spline (smoothing)
  * Bootstrapping                 : par-curve bootstrap with semi-annual coupons

Conventions
  * Input YTMs are % with semi-annual compounding (KTB convention).
  * Internally every curve is a continuously-compounded zero rate r(t).
  * Display rates are converted back to semi-annual compounding (%).
  * NS / NSS / Spline are fitted directly to the YTMs (YTM ~ spot
    approximation, standard for quick curve work); Bootstrapping strips
    coupons from the interpolated par curve, so it is the exact-spot model.

Every model returns a callable r(t) (cont. comp, decimal) for t in years.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MODEL_NAMES = ["Nelson-Siegel", "Svensson", "Cubic Spline", "Bootstrapping"]
FREQ = 2  # semi-annual coupons / compounding


# ---------------------------------------------------------------------------
# Rate conversions
# ---------------------------------------------------------------------------
def sa_to_cc(y_pct: np.ndarray) -> np.ndarray:
    return FREQ * np.log1p(np.asarray(y_pct, float) / (100 * FREQ))


def cc_to_sa(r: np.ndarray) -> np.ndarray:
    return 100 * FREQ * np.expm1(np.asarray(r, float) / FREQ)


@dataclass
class CurveFit:
    name: str
    zero: Callable[[np.ndarray], np.ndarray]          # t -> r(t), cont. comp decimal
    params: dict = field(default_factory=dict)
    fitted_obs: np.ndarray | None = None              # model YTM at obs t (sa %)

    def spot_sa(self, t: np.ndarray) -> np.ndarray:
        return cc_to_sa(self.zero(np.asarray(t, float)))

    def forward_sa(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Discrete forward rate between t1 and t2 (sa %)."""
        t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
        r1 = np.where(t1 > 0, self.zero(np.maximum(t1, 1e-6)), 0.0)
        r2 = self.zero(t2)
        f = (r2 * t2 - r1 * t1) / (t2 - t1)
        return cc_to_sa(f)

    def inst_forward_sa(self, t: np.ndarray, h: float = 1e-3) -> np.ndarray:
        """Instantaneous forward f(t) = d(r t)/dt (sa %)."""
        t = np.asarray(t, float)
        if getattr(self, "inst", None) is not None:        # analytic (monotone convex)
            return cc_to_sa(self.inst(t))
        lo = np.maximum(t - h, 1e-4)
        hi = t + h
        f = (self.zero(hi) * hi - self.zero(lo) * lo) / (hi - lo)
        return cc_to_sa(f)


# ---------------------------------------------------------------------------
# Nelson-Siegel / Svensson
# ---------------------------------------------------------------------------
def _ns_loadings(t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(t, 1e-8) / tau
    l1 = (1 - np.exp(-x)) / x
    l2 = l1 - np.exp(-x)
    return l1, l2


def _ns_design(t: np.ndarray, taus: tuple[float, ...]) -> np.ndarray:
    cols = [np.ones_like(t)]
    l1, l2 = _ns_loadings(t, taus[0])
    cols += [l1, l2]
    for tau in taus[1:]:
        cols.append(_ns_loadings(t, tau)[1])
    return np.column_stack(cols)


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    sse = float(np.sum(w * (X @ beta - y) ** 2))
    return beta, sse


def fit_nelson_siegel(t: np.ndarray, y_pct: np.ndarray, w: np.ndarray | None = None) -> CurveFit:
    r = sa_to_cc(y_pct)
    w = np.ones_like(t) if w is None else w
    best = None
    for tau in np.geomspace(0.1, 30, 120):
        beta, sse = _ols(_ns_design(t, (tau,)), r, w)
        if best is None or sse < best[0]:
            best = (sse, tau, beta)
    _, tau, beta = best

    def zero(tt, beta=beta, tau=tau):
        return _ns_design(np.atleast_1d(np.asarray(tt, float)), (tau,)) @ beta

    fit = CurveFit("Nelson-Siegel", zero,
                   dict(beta0=beta[0], beta1=beta[1], beta2=beta[2], tau=tau))
    fit.fitted_obs = fit.spot_sa(t)
    return fit


def fit_svensson(t: np.ndarray, y_pct: np.ndarray, w: np.ndarray | None = None) -> CurveFit:
    r = sa_to_cc(y_pct)
    w = np.ones_like(t) if w is None else w
    grid = np.geomspace(0.1, 30, 45)
    best = None
    for i, tau1 in enumerate(grid):
        for tau2 in grid[i + 2:]:           # tau2 > tau1, keep loadings separable
            beta, sse = _ols(_ns_design(t, (tau1, tau2)), r, w)
            if abs(beta[2]) > 1 or abs(beta[3]) > 1:   # reject explosive humps
                continue
            if best is None or sse < best[0]:
                best = (sse, tau1, tau2, beta)
    if best is None:
        ns = fit_nelson_siegel(t, y_pct, w)
        ns.name = "Svensson"
        return ns
    _, tau1, tau2, beta = best

    def zero(tt, beta=beta, tau1=tau1, tau2=tau2):
        return _ns_design(np.atleast_1d(np.asarray(tt, float)), (tau1, tau2)) @ beta

    fit = CurveFit("Svensson", zero,
                   dict(beta0=beta[0], beta1=beta[1], beta2=beta[2], beta3=beta[3],
                        tau1=tau1, tau2=tau2))
    fit.fitted_obs = fit.spot_sa(t)
    return fit


# ---------------------------------------------------------------------------
# Cubic spline (natural cubic regression spline, ESL 5.2.1 basis)
# ---------------------------------------------------------------------------
_DEFAULT_KNOTS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def _natural_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    K = len(knots)
    def d(k):
        return (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3) / (knots[-1] - knots[k])
    cols = [np.ones_like(x), x]
    dK1 = d(K - 2)
    for k in range(K - 2):
        cols.append(d(k) - dK1)
    return np.column_stack(cols)


def fit_cubic_spline(t: np.ndarray, y_pct: np.ndarray, w: np.ndarray | None = None,
                     knots: list[float] | None = None, ridge: float = 1e-6) -> CurveFit:
    r = sa_to_cc(y_pct)
    w = np.ones_like(t) if w is None else w
    lo, hi = float(t.min()), float(t.max())
    kn = [k for k in (knots or _DEFAULT_KNOTS) if lo < k < hi]
    kn = np.array(sorted(set([lo] + kn + [hi])))
    # need >= ~3 points between knots; thin knots where data is sparse
    keep = [kn[0]]
    for k in kn[1:]:
        if np.sum((t > keep[-1]) & (t <= k)) >= 3 or k == kn[-1]:
            keep.append(k)
    kn = np.array(keep)
    if len(kn) < 3:
        kn = np.linspace(lo, hi, 3)

    X = _natural_basis(t, kn)
    sw = np.sqrt(w)
    A = X * sw[:, None]
    beta = np.linalg.solve(A.T @ A + ridge * np.eye(X.shape[1]), A.T @ (r * sw))

    def zero(tt, beta=beta, kn=kn):
        return _natural_basis(np.atleast_1d(np.asarray(tt, float)), kn) @ beta

    fit = CurveFit("Cubic Spline", zero, dict(knots=kn.round(2).tolist()))
    fit.fitted_obs = fit.spot_sa(t)
    return fit


# ---------------------------------------------------------------------------
# Bootstrapping (benchmark par bonds -> discrete forwards -> monotone convex)
# ---------------------------------------------------------------------------
BENCH_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def _benchmark_par(t: np.ndarray, y_pct: np.ndarray,
                   w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Par yield at benchmark tenors: weighted local linear fit of the bonds
    within a window around each tenor. Stripping every individual bond turns
    a few bp of YTM noise into hundreds of bp of forward noise, so the market
    convention of bootstrapping benchmark tenors is used instead."""
    w = np.ones_like(t) if w is None else w
    use = w > 0
    t, y_pct, w = t[use], y_pct[use], w[use]
    lo, hi = float(t.min()), float(t.max())
    nodes = [lo] + [b for b in BENCH_TENORS if lo + 0.05 < b < hi - 0.05] + [hi]
    out_t, out_y = [], []
    for b in nodes:
        half = max(0.2, 0.15 * b)
        m = np.abs(t - b) <= half
        while m.sum() < 3 and half < hi:
            half *= 1.5
            m = np.abs(t - b) <= half
        tt, yy, ww = t[m], y_pct[m], w[m]
        if len(tt) >= 2 and np.ptp(tt) > 1e-6:
            _, c = np.polyfit(tt - b, yy, 1, w=np.sqrt(ww))
        else:
            c = float(np.average(yy, weights=ww))
        out_t.append(b); out_y.append(c)
    return np.array(out_t), np.array(out_y)


def _monotone_convex(taus: np.ndarray, fd: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Hagan-West (2006) monotone convex interpolation of instantaneous
    forwards from discrete forwards fd_i on (tau_{i-1}, tau_i], tau_0 = 0.
    The result is continuous, preserves each interval's average forward
    (so the bootstrapped discount factors are kept) and does not overshoot."""
    n = len(fd)
    f = np.empty(n + 1)
    for i in range(1, n):
        a = taus[i] - taus[i - 1]; b = taus[i + 1] - taus[i]
        f[i] = (a * fd[i] + b * fd[i - 1]) / (a + b)
    if n == 1:
        f[0] = f[1] = fd[0]
    else:
        f[0] = fd[0] - 0.5 * (f[1] - fd[0])
        f[n] = fd[n - 1] - 0.5 * (f[n - 1] - fd[n - 1])

    def g_of(x, g0, g1):
        if abs(g0) < 1e-14 and abs(g1) < 1e-14:
            return np.zeros_like(x)
        if (g0 < 0 and -0.5 * g0 <= g1 <= -2 * g0) or (g0 > 0 and -0.5 * g0 >= g1 >= -2 * g0):
            return g0 * (1 - 4 * x + 3 * x ** 2) + g1 * (-2 * x + 3 * x ** 2)
        if (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
            eta = (g1 + 2 * g0) / (g1 - g0)
            return np.where(x <= eta, g0, g0 + (g1 - g0) * ((x - eta) / (1 - eta)) ** 2)
        if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
            eta = 3 * g1 / (g1 - g0)
            return np.where(x < eta, g1 + (g0 - g1) * ((eta - x) / eta) ** 2, g1)
        eta = g1 / (g1 + g0)
        A = -g0 * g1 / (g0 + g1)
        return np.where(x < eta, A + (g0 - A) * ((eta - x) / eta) ** 2,
                        A + (g1 - A) * ((x - eta) / (1 - eta)) ** 2)

    def inst_fwd(t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.empty_like(t)
        idx = np.clip(np.searchsorted(taus, t, side="left"), 1, n)
        for i in np.unique(idx):
            m = idx == i
            x = np.clip((t[m] - taus[i - 1]) / (taus[i] - taus[i - 1]), 0, 1)
            out[m] = fd[i - 1] + g_of(x, f[i - 1] - fd[i - 1], f[i] - fd[i - 1])
        out[t > taus[-1]] = f[n]                    # flat beyond the last node
        return out

    return inst_fwd


def fit_bootstrap(t: np.ndarray, y_pct: np.ndarray, w: np.ndarray | None = None) -> CurveFit:
    t = np.asarray(t, float); y_pct = np.asarray(y_pct, float)
    ut, uy = _benchmark_par(t, y_pct, w)

    # 1) node-by-node bootstrap of discrete forwards. Each benchmark is a
    #    par bond paying semi-annual coupons back from its maturity; coupon
    #    dates inside the unknown last interval use that interval's (flat)
    #    forward, solved by bisection so the bond prices at par.
    taus = np.r_[0.0, ut]
    fd: list[float] = []

    def log_df(s, fds):
        s = np.atleast_1d(s)
        out = np.zeros_like(s)
        for i, f_i in enumerate(fds):
            out -= f_i * np.clip(s - taus[i], 0, taus[i + 1] - taus[i])
        return out

    for i, (T, c_pct) in enumerate(zip(ut, uy)):
        c = c_pct / 100.0
        if T <= 1.0 / FREQ + 1e-9:                  # money-market: zero coupon
            target = -sa_to_cc(c_pct) * T
            fd.append(float((log_df(np.array([taus[i]]), fd)[0] - target) / (T - taus[i])))
            continue
        cpn_t = T - np.arange(0, int(np.ceil(T * FREQ - 1e-9))) / FREQ
        cpn_t = cpn_t[cpn_t > 1e-9]

        def price(f_last):
            ld = log_df(cpn_t, fd + [f_last])
            return float(np.sum(c / FREQ * np.exp(ld)) + np.exp(ld[0]))   # cpn_t[0] == T

        lo_f, hi_f = -0.10, 0.50
        for _ in range(80):
            mid = 0.5 * (lo_f + hi_f)
            if price(mid) > 1.0:
                lo_f = mid
            else:
                hi_f = mid
        fd.append(0.5 * (lo_f + hi_f))
    fd_arr = np.array(fd)

    # 2) continuous instantaneous forwards (monotone convex), integrated to r(t)
    inst = _monotone_convex(taus, fd_arr)
    t_end = max(taus[-1], 60.0)
    fine = np.linspace(0.0, t_end, int(t_end * 365) + 1)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inst(fine[1:]) + inst(fine[:-1])) * np.diff(fine))])

    def zero(tt, fine=fine, cum=cum, inst=inst, t_end=t_end):
        tt = np.atleast_1d(np.asarray(tt, float))
        tc = np.maximum(tt, 1e-6)
        I = np.interp(tc, fine, cum)
        I = np.where(tc > t_end, cum[-1] + inst(np.array([t_end]))[0] * (tc - t_end), I)
        return I / tc

    fit = CurveFit("Bootstrapping", zero,
                   dict(benchmarks=", ".join(f"{a:g}y:{b:.3f}" for a, b in zip(ut, uy))))
    fit.fitted_obs = np.interp(t, ut, uy)           # benchmark par curve at obs t
    fit.inst = inst
    return fit


# ---------------------------------------------------------------------------
# Outliers / on-the-run richness
# ---------------------------------------------------------------------------
def robust_weights(t: np.ndarray, y_pct: np.ndarray, mode: str = "none",
                   thresh_bp: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Weights from residuals to a smooth reference curve (Svensson).

    mode="none"     : all 1
    mode="exclude"  : 0 for |residual| > thresh_bp, else 1 (two passes)
    mode="downweight": Huber weights min(1, thresh/|res|) (iterated)
    Returns (weights, residual_bp vs the final reference fit).
    """
    t = np.asarray(t, float); y = np.asarray(y_pct, float)
    w = np.ones_like(t)
    res = np.zeros_like(t)
    if mode == "none":
        res = (y - fit_svensson(t, y).fitted_obs) * 100
        return w, res
    for _ in range(3):
        ref = fit_svensson(t, y, w)
        res = (y - ref.fitted_obs) * 100
        if mode == "exclude":
            w_new = (np.abs(res) <= thresh_bp).astype(float)
        else:
            w_new = np.minimum(1.0, thresh_bp / np.maximum(np.abs(res), 1e-9))
        if np.allclose(w_new, w):
            break
        w = w_new
    return w, res


# ---------------------------------------------------------------------------
def fit_all(t: np.ndarray, y_pct: np.ndarray, w: np.ndarray | None = None) -> dict[str, CurveFit]:
    t = np.asarray(t, float); y = np.asarray(y_pct, float)
    return {
        "Nelson-Siegel": fit_nelson_siegel(t, y, w),
        "Svensson":      fit_svensson(t, y, w),
        "Cubic Spline":  fit_cubic_spline(t, y, w),
        "Bootstrapping": fit_bootstrap(t, y, w),
    }
