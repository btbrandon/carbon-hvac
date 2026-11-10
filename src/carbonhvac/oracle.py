"""Perfect-foresight upper bound by finite-horizon backward induction (value iteration).

Given the realised exogenous series for the episode's day (weather, true occupancy,
price, CI), the dynamics are deterministic, so the optimal cost-to-go on a discretised
indoor-temperature grid is

    V_H(T) = 0,   V_t(T) = min_a [ c_t(T, a) + V_{t+1}(f_t(T, a)) ]

with V_{t+1} linearly interpolated between grid points. At execution time the action is
chosen by a one-step lookahead from the exact current temperature, so the only
approximation is the interpolation of V.
"""
import numpy as np

from .data import STEPS_PER_DAY
from .env import discomfort, simulate_interval, step_cost


class Oracle:
    name = "oracle"

    def __init__(self, t_min=15.0, t_max=35.0, resolution=0.05):
        self.grid = np.arange(t_min, t_max + 1e-9, resolution)

    def _q(self, env, k, t_in, v_next):
        cfg, n_sub = env.cfg, env.n_sub
        idx0 = env.day * STEPS_PER_DAY + k * n_sub
        occ = env.occ_true[k * n_sub:(k + 1) * n_sub]
        q = []
        for a in range(cfg.n_actions):
            t_end, e, c, d, devs = simulate_interval(cfg, env.prof, idx0, n_sub, t_in, a)
            cost = step_cost(cfg, e, c, d, discomfort(cfg, devs, occ))
            q.append(cost + np.interp(t_end, self.grid, v_next))
        return np.stack(q)

    def plan(self, env):
        v = np.zeros((env.horizon + 1, len(self.grid)))
        for k in reversed(range(env.horizon)):
            v[k] = self._q(env, k, self.grid, v[k + 1]).min(axis=0)
        self.v = v

    def __call__(self, obs, env):
        if env.k == 0:  # new episode: replan on its realised exogenous series
            self.plan(env)
        return int(np.argmin(self._q(env, env.k, np.array(env.t_in), self.v[env.k + 1])))

    def optimal_cost(self, env):
        """Optimal total cost from the episode's initial state (call right after reset)."""
        self.plan(env)
        return float(np.interp(env.t_in, self.grid, self.v[0]))
