"""Simulator validation (§5): step response, conservation/bounds, energy magnitudes, oracle optimality."""
import numpy as np
import pytest

from carbonhvac.baselines import CarbonHeuristic, FixedSetpoint, OccupancySetback
from carbonhvac.data import STEPS_PER_DAY, synthetic_profiles
from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import eval_episodes, run_episode
from carbonhvac.oracle import Oracle
from carbonhvac.thermal import ThermalParams, step_temperature


@pytest.fixture(scope="module")
def env():
    return CarbonHVACEnv(EnvConfig(split="test"))


def test_step_response_time_constant():
    """Free-floating zone after a 10 K outdoor step reaches 63.2% of the step after tau hours."""
    p = ThermalParams()
    t, dt = 20.0, 0.25
    for _ in range(int(p.tau / dt)):
        t = step_temperature(p, t, 30.0, 0.0, 0.0, dt)
    assert t == pytest.approx(20 + 10 * (1 - np.exp(-1)), abs=1e-6)


def test_exact_discretisation_is_step_size_invariant():
    p = ThermalParams()
    one = step_temperature(p, 25.0, 31.0, 10.0, 7.0, 1.0)
    four = 25.0
    for _ in range(4):
        four = step_temperature(p, four, 31.0, 10.0, 7.0, 0.25)
    assert one == pytest.approx(four, abs=1e-9)


def test_more_cooling_means_colder(env):
    env.reset(options={"day": 280, "noise_seed": 0})
    preds = [env.predict_next_t(a) for a in range(env.action_space.n)]
    assert all(np.diff(preds) < 0)


def test_accounting_identities(env):
    trace = run_episode(env, FixedSetpoint(), 280, 0)
    assert len(trace) == STEPS_PER_DAY
    assert (trace["energy_kwh"] >= 0).all()
    assert np.allclose(trace["carbon_kg"], trace["energy_kwh"] * trace["ci"])
    assert np.allclose(trace["cost_sgd"], trace["energy_kwh"] * trace["price"])
    assert trace["energy_kwh"].max() <= ThermalParams().p_max * 0.25 + 1e-9
    assert (trace.loc[~trace["occupied"], "comfort_dh"] == 0).all()


def test_energy_magnitude_is_plausible():
    """Cooling electricity per m2-year should be on the order of a Singapore office's ACMV use.

    Typical SG office EUI is ~200 kWh/m2/yr with ACMV ~50-60% of it; for a 500 m2 zone running
    only on office hours we accept 40-200 kWh/m2/yr.
    """
    env = CarbonHVACEnv(EnvConfig(split="all"))
    days = range(0, 365, 7)
    kwh = sum(run_episode(env, OccupancySetback(), d, 0)["energy_kwh"].sum() for d in days)
    per_m2_year = kwh / len(days) * 365 / 500
    assert 40 < per_m2_year < 200, per_m2_year


def test_setback_keeps_comfort(env):
    trace = run_episode(env, OccupancySetback(), 280, 0)
    occupied = trace[trace["occupied"]]
    assert (occupied["t_in"] < 26.0 + 0.3).all()


def test_ci_and_price_not_identical():
    p = synthetic_profiles()
    r = np.corrcoef(p.price, p.ci)[0, 1]
    assert -0.9 < r < 0.9


def test_oracle_dominates_baselines(env):
    """The DP optimum must have reward >= every baseline on the same episode (up to grid error)."""
    for day, ns in eval_episodes("test", n=4, seed=0):
        oracle_r = run_episode(env, Oracle(), day, ns)["reward"].sum()
        for ctrl in (FixedSetpoint(), OccupancySetback(), CarbonHeuristic()):
            r = run_episode(env, ctrl, day, ns)["reward"].sum()
            assert oracle_r >= r - 0.05, (day, ctrl.name, oracle_r, r)


def test_oracle_value_matches_rollout(env):
    env.reset(options={"day": 290, "noise_seed": 3})
    predicted = -Oracle().optimal_cost(env)
    realised = run_episode(env, Oracle(), 290, 3)["reward"].sum()
    assert realised == pytest.approx(predicted, rel=0.02, abs=0.1)


def test_hourly_control_runs():
    env = CarbonHVACEnv(EnvConfig(split="test", control_minutes=60))
    trace = run_episode(env, OccupancySetback(), 280, 0)
    assert len(trace) == 24
