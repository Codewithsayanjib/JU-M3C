"""Bootstrap CIs and paired significance tests. All resampling is over notes."""
import numpy as np
from scipy.stats import binomtest

N_BOOT = 1000
SEED = 42


def bootstrap_ci(stat, n: int, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05):
    """stat(idx) -> float for an index array. Returns (point, lo, hi) percentile CI."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mean_ci(values, **kw):
    v = np.asarray(values, dtype=float)
    return bootstrap_ci(lambda i: v[i].mean(), len(v), **kw)


def median_ci(values, **kw):
    v = np.asarray(values, dtype=float)
    return bootstrap_ci(lambda i: float(np.median(v[i])), len(v), **kw)


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar (binomial on discordant pairs) for two classifiers on the same notes."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())  # A right, B wrong
    n10 = int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"n": len(a), "a_only": n01, "b_only": n10, "p_value": float(p)}


def paired_bootstrap(scores_a, scores_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean difference A-B over the same notes, 95% CI and two-sided bootstrap p-value."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"n": len(d), "mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(p)}
