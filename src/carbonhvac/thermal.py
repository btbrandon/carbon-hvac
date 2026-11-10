"""Single-zone 1R1C thermal model of an office floor with a cooling-only HVAC unit.

    C dT/dt = (T_out - T_in) / R + Q_gain - Q_cool,   Q_cool = COP(T_out) * P_elec

Units: temperature in degC, power in kW, energy in kWh, C in kWh/K, R in K/kW, time in hours.
Inputs are held constant over a step, so we use the exact exponential discretisation
rather than forward Euler (stable for any dt).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalParams:
    R: float = 0.4            # envelope resistance, K/kW  (~500 m2 office floor)
    C: float = 15.0           # thermal capacitance, kWh/K
    p_max: float = 14.0       # max electrical HVAC power, kW
    cop_nominal: float = 3.5  # COP at 30 degC outdoor
    cop_slope: float = 0.08   # COP drop per K of outdoor temperature above 30 degC
    cop_min: float = 2.5
    cop_max: float = 4.5

    @property
    def tau(self) -> float:
        """Time constant RC in hours."""
        return self.R * self.C


def cop(params: ThermalParams, t_out):
    return np.clip(params.cop_nominal - params.cop_slope * (t_out - 30.0),
                   params.cop_min, params.cop_max)


def step_temperature(params: ThermalParams, t_in, t_out, q_gain, p_elec, dt):
    """Indoor temperature after dt hours. Vectorised over any argument."""
    q_cool = cop(params, t_out) * p_elec
    t_eq = t_out + params.R * (q_gain - q_cool)
    return t_eq + (t_in - t_eq) * np.exp(-dt / params.tau)
