"""Statistics for the §7 protocol: bootstrap CIs, Welch tests, Holm-Bonferroni."""
import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot=10_000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def diff_ci(a, b, n_boot=10_000, alpha=0.05, seed=0):
    """Bootstrap CI of mean(a) - mean(b) for independent samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = (rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1))
    return a.mean() - b.mean(), np.quantile(d, alpha / 2), np.quantile(d, 1 - alpha / 2)


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj
