"""HVAC equipment component models.

This module contains physical models for HVAC equipment components including
valves, pumps, chillers, heat exchangers, cooling towers, fans, and humidifiers.

References:
    空気調和・衛生工学会編：空気調和・衛生工学便覧14版，1基礎編，2010.
    宇田川光弘：パソコンによる空気調和計算法，オーム社，1986.
    EnergyPlus Engineering Reference (22.1), Variable Refrigerant Flow Heat Pumps.
"""

from __future__ import annotations

import math
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

from .heat_exchanger import getparameter_hex, hex_effectiveness, hex_ntu
from .psychrometrics import (
    h_rh2w,
    pv2w,
    tdb2hsat,
    tdb_rh2h_x,
    tdb_rh2twb,
    tdb_twb2w,
    tdb_w2h,
    tdp2psat,
    w2cpair,
    w_h2tdb,
    w_rh2tdb,
    w_tdb2rh,
)


def quadratic_formula(co_0: float, co_1: float, co_2: float) -> list[float, int]:
    """Solve the quadratic equation co_2*x^2 + co_1*x + co_0 = 0 for positive x.

    Returns the larger non-negative root. Used for pump/fan curve intersections.

    Args:
        co_0: Constant term coefficient.
        co_1: Linear term coefficient.
        co_2: Quadratic term coefficient.

    Returns:
        A list ``[g, flag]`` where *g* is the positive root (0.0 if none) and
        *flag* is 0 on success, 1 if both roots are negative, 2 if the
        discriminant is non-positive.
    """
    flag = 0
    if co_1 ** 2 - 4 * co_2 * co_0 > 0:
        g1 = (-co_1 + (co_1 ** 2 - 4 * co_2 * co_0) ** 0.5) / (2 * co_2)
        g2 = (-co_1 - (co_1 ** 2 - 4 * co_2 * co_0) ** 0.5) / (2 * co_2)
        if max(g1, g2) < 0:
            g = 0.0
            flag = 1
        else:
            g = max(g1, g2)
    else:
        g = 0.0
        flag = 2
    return [g, flag]


# ---------------------------------------------------------------------------
# Valve
# ---------------------------------------------------------------------------

class Valve:
    """Equal-percentage valve with pressure-drop / flow-rate characteristic.

    Attributes:
        cv_max: Maximum flow coefficient Cv.
        r: Rangeability (inverse of minimum flow ratio at fully closed).
        vlv: Valve opening (0.0 = fully closed, 1.0 = fully open).
        dp: Pressure drop across valve [kPa].
        g: Flow rate [m³/min].
    """

    def __init__(self, cv_max: float = 800, r: float = 100) -> None:
        """Initialize valve with rating parameters.

        Args:
            cv_max: Maximum Cv (flow coefficient).
            r: Rangeability; default 100.
        """
        self.cv_max = cv_max
        self.r = r
        self.dp = 0.0
        self.vlv = 0.0
        self.g = 0.0

    def f2p(self, g: float) -> float:
        """Calculate pressure drop from flow rate.

        Args:
            g: Flow rate [m³/min].

        Returns:
            Pressure drop [kPa] (negative convention).
        """
        self.g = g
        if self.vlv == 0.0:
            self.dp = -99999999
        elif (self.cv_max * self.r ** (self.vlv - 1)) ** 2 > 0:
            self.dp = (
                -1743 * (1000 / 60) ** 2
                / (self.cv_max * self.r ** (self.vlv - 1)) ** 2
            ) * self.g ** 2
        else:
            self.dp = 0.0
        if self.g < 0:
            self.dp = -self.dp
        return self.dp

    def p2f(self, dp: float) -> float:
        """Calculate flow rate from pressure drop.

        Args:
            dp: Pressure drop [kPa].

        Returns:
            Flow rate [m³/min].
        """
        self.dp = dp
        if self.vlv == 0.0:
            self.g = 0
        elif self.dp < 0:
            self.g = (
                self.dp
                / (-1743 * (1000 / 60) ** 2 / (self.cv_max * self.r ** (self.vlv - 1)) ** 2)
            ) ** 0.5
        else:
            self.g = -(
                self.dp
                / (1743 * (1000 / 60) ** 2 / (self.cv_max * self.r ** (self.vlv - 1)) ** 2)
            ) ** 0.5
        return self.g

    def f2p_co(self) -> np.ndarray:
        """Return polynomial coefficients [c0, c1, c2] for dp = c0 + c1*g + c2*g².

        Returns:
            Array of coefficients ``[0, 0, c2]``.
        """
        if self.vlv == 0.0:
            return np.array([0, 0, -99999999])
        return np.array(
            [0, 0, -1743 * (1000 / 60) ** 2 / (self.cv_max * self.r ** (self.vlv - 1)) ** 2]
        )


# ---------------------------------------------------------------------------
# Pump
# ---------------------------------------------------------------------------

class Pump:
    """Centrifugal pump with pressure-flow and efficiency curves.

    Supports INV (inverter) speed control via the *inv* attribute.

    Attributes:
        pg: Pressure-flow curve coefficients [c0, c1, c2].
        eg: Efficiency-flow curve coefficients [c0, c1, c2].
        r_ef: Rated maximum efficiency.
        inv: Inverter frequency ratio (0.0–1.0).
        g_d: Rated flow rate [m³/min].
        dp: Pump head [kPa].
        g: Flow rate [m³/min].
        ef: Pump efficiency.
        pw: Power consumption [kW].
        flag: 0 = normal, 1 = head clipped to 0, 2 = zero efficiency.
        para: Non-zero if operating in parallel mode.
    """

    def __init__(
        self,
        pg: list[float] | None = None,
        eg: list[float] | None = None,
        r_ef: float = 0.8,
        g_d: float = 0.25,
        inv: float = 1.0,
        figure: int = 1,
    ) -> None:
        """Initialize pump with performance curve parameters.

        Args:
            pg: Pressure-flow curve coefficients [c0, c1, c2].
                Default: ``[233, 5.9578, -4.95]``.
            eg: Efficiency-flow curve coefficients [c0, c1, c2].
                Default: ``[0.0099, 0.4174, -0.0508]``.
            r_ef: Rated maximum efficiency. Default: 0.8.
            g_d: Rated flow rate [m³/min]. Default: 0.25.
            inv: INV frequency ratio (0.0–1.0). Default: 1.0.
            figure: If 1, display the performance curve on instantiation.
        """
        (_, _, _, text) = traceback.extract_stack()[-2]
        self.name = text[: text.find("=")].strip()
        self.pg = [233, 5.9578, -4.95] if pg is None else pg
        self.eg = [0.0099, 0.4174, -0.0508] if eg is None else eg
        self.r_ef = r_ef
        self.inv = inv
        self.g_d = g_d
        self.dp = 0.0
        self.g = 0.0
        self.ef = 0.0
        self.pw = 0.0
        self.flag = 0
        self.para = 0
        if figure == 1:
            self.figure_curve()

    def f2p(self, g: float) -> float:
        """Calculate pump head from flow rate (affinity laws applied).

        Args:
            g: Flow rate [m³/min].

        Returns:
            Pump head [kPa].
        """
        self.g = g
        if self.g > 0 and self.inv > 0:
            self.dp = (
                self.pg[0]
                + self.pg[1] * (self.g / self.inv)
                + self.pg[2] * (self.g / self.inv) ** 2
            ) * self.inv ** 2
        else:
            self.dp = 0.0
        if self.dp < 0:
            self.dp = 0.0
            self.flag = 1
        else:
            self.flag = 0
        return self.dp

    def f2p_co(self) -> list[float]:
        """Return polynomial coefficients adjusted for current INV ratio.

        Returns:
            Coefficients ``[c0, c1, c2]`` such that dp = c0 + c1*g + c2*g².
        """
        return [self.pg[0] * self.inv ** 2, self.pg[1] * self.inv, self.pg[2]]

    def cal(self) -> None:
        """Calculate power consumption and efficiency at the current operating point.

        Updates *pw*, *ef*, *dp*, and *flag* in place.
        """
        if self.g > 0 and self.inv > 0:
            g = self.g / self.inv
            k = (1.0 - (1.0 - self.r_ef) / (self.inv ** 0.2)) / self.r_ef
            self.ef = k * (self.eg[0] + self.eg[1] * g + self.eg[2] * g ** 2)
            self.dp = (self.pg[0] + self.pg[1] * g + self.pg[2] * g ** 2) * self.inv ** 2
            if self.dp < 0:
                self.dp = 0.0
                self.flag = 1
            if self.ef > 0:
                self.pw = 1.0 * self.g * self.dp / (60 * self.ef)
                self.flag = 0
            else:
                self.pw = 0.0
                self.flag = 2
        else:
            self.pw = 0.0
            self.ef = 0.0
            self.flag = 0

    def figure_curve(self) -> None:
        """Display pressure-flow and efficiency-flow curves using matplotlib."""
        quadratic_formula(self.pg[0], self.pg[1], self.pg[2])
        x = np.linspace(0, self.g_d, 50)
        y_p = (
            self.pg[0]
            + self.pg[1] * (x / self.inv)
            + self.pg[2] * (x / self.inv) ** 2
        ) * self.inv ** 2
        x2 = np.linspace(0, self.g_d * self.inv, 50)
        k = (1.0 - (1.0 - self.r_ef) / (self.inv ** 0.2)) / self.r_ef
        y_e = k * (self.eg[0] + self.eg[1] * (x2 / self.inv) + self.eg[2] * (x2 / self.inv) ** 2)
        fig, ax1 = plt.subplots()
        color1 = "tab:orange"
        ax1.set_xlabel("Flow [m3/min]")
        ax1.set_ylabel("Pressure [kPa]", color=color1)
        ax1.plot(x, y_p, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, self.pg[0] + 10)
        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("Efficiency [-]", color=color2)
        ax2.plot(x2, y_e, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 1)
        plt.title("{}".format(self.name))
        plt.show()


# ---------------------------------------------------------------------------
# Chiller
# ---------------------------------------------------------------------------

class Chiller:
    """Water-cooled chiller modeled by a COP look-up table.

    Equipment specifications and COP data are read from an Excel file.
    The table rows represent cooling-water inlet temperatures and columns
    represent part-load ratios (PLR).

    Attributes:
        tout_ch_d: Rated chilled-water outlet temperature [°C].
        tin_ch_d: Rated chilled-water inlet temperature [°C].
        g_ch_d: Rated chilled-water flow rate [m³/min].
        q_ch_d: Rated cooling capacity [kW].
        pw_d: Rated motor input power [kW].
        kr_ch: Evaporator pressure-loss coefficient [kPa/(m³/min)²].
        kr_cd: Condenser pressure-loss coefficient [kPa/(m³/min)²].
        cop: Current COP.
        pl: Current part-load ratio.
        pw: Current power consumption [kW].
        flag: 0 = normal; 1–5 = out-of-range warning codes.
    """

    def __init__(
        self,
        filename: str = "equipment_spec.xlsx",
        sheet_name: str = "Chiller",
    ) -> None:
        """Load chiller specifications from an Excel file.

        Args:
            filename: Path to the Excel specification file.
            sheet_name: Sheet name containing chiller data.
        """
        spec_table = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        self.tout_ch_d = float(spec_table.iat[1, 0])
        self.tin_ch_d = float(spec_table.iat[1, 1])
        self.g_ch_d = float(spec_table.iat[1, 2])
        self.tin_cd_d = float(spec_table.iat[1, 3])
        self.tout_cd_d = float(spec_table.iat[1, 4])
        self.g_cd_d = float(spec_table.iat[1, 5])
        self.q_ch_d = (self.tin_ch_d - self.tout_ch_d) * self.g_ch_d * 1000 * 4.186 / 60
        self.pw_d = float(spec_table.iat[1, 6])
        self.kr_ch = float(spec_table.iat[1, 7])
        self.kr_cd = float(spec_table.iat[1, 8])
        self.pw_sub = 0
        self.COP_rp = self.q_ch_d / self.pw_d
        self.tout_ch = 7.0
        self.tout_cd = 37.0
        self.g_ch = 0.0
        self.g_cd = 0.0
        self.pw = 0.0
        self.q_ch = 0.0
        self.pl = 0.0
        self.cop = 0.0
        self.flag = 0
        self.dp_ch = 0.0
        self.dp_cd = 0.0
        self.tin_cd = 15.0
        self.tin_ch = 7.0
        self.tout_ch_sp = 7.0

        pl_cop = spec_table.drop(spec_table.index[[0, 1, 2]])
        pl_cop.iat[0, 0] = "-"
        pl_cop = pl_cop.dropna(how="all", axis=1)
        self.data = pl_cop.values

    def cal(
        self,
        tout_ch_sp: float,
        tin_ch: float,
        g_ch: float,
        tin_cd: float,
        g_cd: float,
    ) -> None:
        """Calculate chiller performance at the current operating point.

        Args:
            tout_ch_sp: Chilled-water outlet temperature setpoint [°C].
            tin_ch: Chilled-water inlet temperature [°C].
            g_ch: Chilled-water flow rate [m³/min].
            tin_cd: Cooling-water inlet temperature [°C].
            g_cd: Cooling-water flow rate [m³/min].
        """
        self.flag = 0
        self.tout_ch_sp = tout_ch_sp
        self.tin_ch = tin_ch
        self.g_ch = g_ch
        self.tin_cd = tin_cd
        self.g_cd = g_cd
        self.tout_ch = self.tout_ch_sp
        self.q_ch = (self.tin_ch - self.tout_ch) * self.g_ch * 1000 * 4.186 / 60

        if self.q_ch > 0 and self.g_cd > 0:
            pl = self.data[0][1:].astype(np.float32)
            temp = self.data.transpose()[0][1:].astype(np.float32)
            dataset = self.data[1:].transpose()[1:].transpose().astype(np.float32)
            cop = RegularGridInterpolator((temp, pl), dataset)

            self.pl = self.q_ch / self.q_ch_d
            pl_cop = self.pl
            if self.pl > pl[-1]:
                self.pl = pl[-1]
                pl_cop = self.pl
                self.tout_ch += (self.q_ch - self.q_ch_d) / (self.g_ch * 1000 * 4.186 / 60)
                self.q_ch = self.q_ch_d
                self.flag = 1
            elif self.pl < pl[0]:
                pl_cop = pl[-1]
                self.flag = 2

            tin_cd_cop = self.tin_cd - (self.tout_ch - self.tout_ch_d)
            if tin_cd_cop < temp[0]:
                tin_cd_cop = temp[0]
                self.flag = 3
            elif tin_cd_cop > temp[-1]:
                tin_cd_cop = temp[-1]
                self.flag = 4

            self.cop = float(cop([[tin_cd_cop, pl_cop]]))
            self.pw = self.q_ch / self.cop + self.pw_sub
            self.tout_cd = (
                (self.q_ch + self.pw) / (4.186 * self.g_cd * 1000 / 60) + self.tin_cd
            )
        elif self.q_ch == 0:
            self.pw = 0.0
            self.cop = 0.0
            self.pl = 0.0
            self.flag = 0
        else:
            self.pw = 0.0
            self.cop = 0.0
            self.pl = 0.0
            self.flag = 5

        self.dp_ch = -self.kr_ch * g_ch ** 2
        self.dp_cd = -self.kr_cd * g_cd ** 2


# ---------------------------------------------------------------------------
# AirSourceHeatPump
# ---------------------------------------------------------------------------

class AirSourceHeatPump:
    """Air-source heat pump (cooling mode) modeled by a COP look-up table.

    Table rows represent outdoor dry-bulb temperatures; columns represent PLR.

    Attributes:
        q_ch_d: Rated cooling capacity [kW].
        pw_d: Rated motor input power [kW].
        kr_ch: Chilled-water pressure-loss coefficient [kPa/(m³/min)²].
        cop: Current COP.
        pl: Current part-load ratio.
        pw: Current power consumption [kW].
        flag: 0 = normal; 1–5 = out-of-range warning codes.
    """

    def __init__(
        self,
        filename: str = "equipment_spec.xlsx",
        sheet_name: str = "AirSourceHeatPump",
    ) -> None:
        """Load air-source heat pump specifications from an Excel file.

        Args:
            filename: Path to the Excel specification file.
            sheet_name: Sheet name containing ASHP data.
        """
        spec_table = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        self.tout_ch_d = float(spec_table.iat[1, 0])
        self.tin_ch_d = float(spec_table.iat[1, 1])
        self.g_ch_d = float(spec_table.iat[1, 2])
        self.q_ch_d = (self.tin_ch_d - self.tout_ch_d) * self.g_ch_d * 1000 * 4.186 / 60
        self.pw_d = float(spec_table.iat[1, 3])
        self.kr_ch = float(spec_table.iat[1, 4])
        self.pw_sub = 0.0
        self.COP_rp = self.q_ch_d / self.pw_d
        self.tout_ch = 7.0
        self.pw = 0.0
        self.q_ch = 0.0
        self.pl = 0.0
        self.cop = 0.0
        self.flag = 0
        self.dp_ch = 0.0
        self.tin_ch = 15.0
        self.g_ch = 0.0
        self.tout_ch_sp = 7.0
        self.tdb = 25.0

        pl_cop = spec_table.drop(spec_table.index[[0, 1, 2]])
        pl_cop.iat[0, 0] = "-"
        pl_cop = pl_cop.dropna(how="all", axis=1)
        self.data = pl_cop.values

    def cal(
        self,
        tout_ch_sp: float,
        tin_ch: float,
        g_ch: float,
        tdb: float,
    ) -> None:
        """Calculate ASHP performance at the current operating point.

        Args:
            tout_ch_sp: Chilled-water outlet temperature setpoint [°C].
            tin_ch: Chilled-water inlet temperature [°C].
            g_ch: Chilled-water flow rate [m³/min].
            tdb: Outdoor dry-bulb temperature [°C].
        """
        self.tdb = tdb
        self.flag = 0
        self.tout_ch_sp = tout_ch_sp
        self.tin_ch = tin_ch
        self.g_ch = g_ch
        self.tout_ch = self.tout_ch_sp
        self.q_ch = (self.tin_ch - self.tout_ch) * self.g_ch * 1000 * 4.186 / 60

        if self.q_ch > self.q_ch_d:
            self.tout_ch = self.tout_ch_sp + (self.q_ch - self.q_ch_d) / (
                self.g_ch * 1000 * 4.186 / 60
            )
            self.q_ch = self.q_ch_d
            self.flag = 1

        if self.q_ch > 0:
            pl = self.data[0][1:].astype(np.float32)
            temp = self.data.transpose()[0][1:].astype(np.float32)
            dataset = self.data[1:].transpose()[1:].transpose().astype(np.float32)
            cop = RegularGridInterpolator((temp, pl), dataset)

            self.pl = self.q_ch / self.q_ch_d
            pl_cop = self.pl
            if self.pl > pl[-1]:
                self.pl = pl[-1]
                pl_cop = self.pl
                self.q_ch = self.q_ch_d
                self.flag = 2
            elif self.pl < pl[0]:
                pl_cop = pl[-1]
                self.flag = 3

            self.cop = float(cop([[tdb, pl_cop]]))
            self.cop *= (
                (273.15 + self.tout_ch) / (tdb - self.tout_ch)
            ) / (
                (273.15 + self.tout_ch_d) / (tdb - self.tout_ch_d)
            )
            self.pw = self.q_ch / self.cop + self.pw_sub
        elif self.q_ch == 0:
            self.pw = 0.0
            self.cop = 0.0
            self.pl = 0.0
            self.flag = 4
        else:
            self.pw = 0.0
            self.cop = 0.0
            self.pl = 0.0
            self.flag = 5

        self.dp_ch = -self.kr_ch * g_ch ** 2


# ---------------------------------------------------------------------------
# AbsorptionChillerESS
# ---------------------------------------------------------------------------

class AbsorptionChillerESS:
    """Absorption chiller/heater modeled per the Japanese Energy-Saving Standard.

    Fuel is assumed to be city gas 13A (lower heating value 40.6 MJ/m³N).

    Attributes:
        rated_capacity_c: Rated cooling capacity [kW].
        rated_input_fuel_c: Rated cooling fuel consumption [Nm³].
        power_c: Rated cooling electricity consumption [kW].
        rated_capacity_h: Rated heating capacity [kW].
        rated_input_fuel_h: Rated heating fuel consumption [Nm³].
        power_h: Rated heating electricity consumption [kW].
        capacity_c: Current cooling output [kW].
        input_fuel_c: Current cooling fuel consumption [Nm³].
        cop_c: Current cooling COP.
        tout_ch: Chilled-water outlet temperature [°C].
        capacity_h: Current heating output [kW].
        input_fuel_h: Current heating fuel consumption [Nm³].
        cop_h: Current heating COP.
        tout_h: Hot-water outlet temperature [°C].
    """

    def __init__(
        self,
        rated_capacity_c: float,
        rated_input_fuel_c: float,
        power_c: float,
        rated_capacity_h: float,
        rated_input_fuel_h: float,
        power_h: float,
    ) -> None:
        """Initialize absorption chiller with rated performance values.

        Args:
            rated_capacity_c: Rated cooling capacity [kW].
            rated_input_fuel_c: Rated cooling fuel consumption [Nm³].
            power_c: Rated cooling electricity consumption [kW].
            rated_capacity_h: Rated heating capacity [kW].
            rated_input_fuel_h: Rated heating fuel consumption [Nm³].
            power_h: Rated heating electricity consumption [kW].
        """
        self.rated_capacity_c = rated_capacity_c
        self.rated_input_fuel_c = rated_input_fuel_c
        self.power_c = power_c
        self.rated_capacity_h = rated_capacity_h
        self.rated_input_fuel_h = rated_input_fuel_h
        self.power_h = power_h

        self.cw_c = 4.192   # specific heat of water at 10 °C [kJ/(kg·K)]
        self.cw_h = 4.178   # specific heat of water at 40 °C [kJ/(kg·K)]
        self.rho_c = 999.741  # density of water at 10 °C [kg/m³]
        self.rho_h = 992.210  # density of water at 40 °C [kg/m³]
        self.cg = 40.6      # lower heating value of city gas 13A [MJ/m³N]
        self.k = 3.6        # unit conversion MJ→kWh [MJ/kWh]

        self.capacity_c = 0.0
        self.input_fuel_c = 0.0
        self.cop_c = 0.0
        self.tout_ch = 7.0

        self.capacity_h = 0.0
        self.input_fuel_h = 0.0
        self.cop_h = 0.0
        self.tout_h = 45.0

    def cal_c(
        self,
        g: float,
        tin_cd: float = 32.0,
        tin_ch: float = 15.0,
        tout_ch_sp: float = 7.0,
    ) -> tuple[float, float, float, float, float]:
        """Calculate cooling-mode performance.

        Args:
            g: Chilled-water flow rate [m³/min].
            tin_cd: Cooling-water inlet temperature [°C]. Default: 32.
            tin_ch: Chilled-water inlet temperature [°C]. Default: 15.
            tout_ch_sp: Chilled-water outlet setpoint [°C]. Default: 7.

        Returns:
            Tuple of ``(capacity_c, input_fuel_c, cop_c, tout_ch, power_c)``.
        """
        k_1 = 1.0
        capacity = self.rated_capacity_c * k_1
        g = g * self.rho_c / 60  # m³/min → kg/s
        self.capacity_c = g * (tin_ch - tout_ch_sp) * self.cw_c

        self.tout_ch = tout_ch_sp
        if self.capacity_c > self.rated_capacity_c:
            delta_t = (self.capacity_c - self.rated_capacity_c) / (g * self.cw_c)
            self.tout_ch = tout_ch_sp + delta_t  # Bug fix: was `tout_chsp` (undefined local)
            self.capacity_c = self.rated_capacity_c

        plr = self.capacity_c / capacity
        plr = max(0.2, min(1.0, plr))

        k_2 = 0.012333 * tin_cd + 0.605333
        k_3 = 0.167757 * plr ** 2 + 0.757814 * plr + 0.074429
        k_4 = -0.01276 * self.tout_ch + 1.0893
        self.input_fuel_c = self.rated_input_fuel_c * k_2 * k_3 * k_4

        q_gas = self.input_fuel_c * self.cg / self.k
        self.cop_c = self.capacity_c / (q_gas + self.power_c)

        return self.capacity_c, self.input_fuel_c, self.cop_c, self.tout_ch, self.power_c

    def cal_h(
        self,
        g: float,
        tin_h: float = 37.0,
        tout_h_sp: float = 45.0,
    ) -> tuple[float, float, float, float, float]:
        """Calculate heating-mode performance.

        Args:
            g: Hot-water flow rate [m³/min].
            tin_h: Hot-water inlet temperature [°C]. Default: 37.
            tout_h_sp: Hot-water outlet setpoint [°C]. Default: 45.

        Returns:
            Tuple of ``(capacity_h, input_fuel_h, cop_h, tout_h, power_h)``.
        """
        k_1 = 1.0
        capacity = self.rated_capacity_h * k_1
        g = g * self.rho_h / 60
        self.capacity_h = g * (tout_h_sp - tin_h) * self.cw_h

        self.tout_h = tout_h_sp
        if self.capacity_h > self.rated_capacity_h:
            delta_t = (self.capacity_h - self.rated_capacity_h) / (g * self.cw_h)
            self.tout_h = tout_h_sp - delta_t
            self.capacity_h = self.rated_capacity_h

        plr = self.capacity_h / capacity
        plr = max(0.1, min(1.0, plr))

        k_2 = 1.0
        k_3 = 1.0 * plr
        k_4 = 1.0
        self.input_fuel_h = self.rated_input_fuel_h * k_2 * k_3 * k_4
        q_gas = self.input_fuel_h * self.cg / self.k
        self.cop_h = self.capacity_h / (q_gas + self.power_h)

        return self.capacity_h, self.input_fuel_h, self.cop_h, self.tout_h, self.power_h


# ---------------------------------------------------------------------------
# VariableRefrigerantFlowESS
# ---------------------------------------------------------------------------

class VariableRefrigerantFlowESS:
    """VRF system modeled per the Japanese Energy-Saving Standard.

    Attributes:
        rated_capacity_c: Rated cooling capacity [kW].
        rated_input_power_c: Rated cooling power input [kW].
        rated_capacity_h: Rated heating capacity [kW].
        rated_input_power_h: Rated heating power input [kW].
        capacity_c: Current cooling capacity [kW].
        input_power_c: Current cooling power [kW].
        cop_c: Current cooling COP.
        capacity_h: Current heating capacity [kW].
        input_power_h: Current heating power [kW].
        cop_h: Current heating COP.
    """

    def __init__(
        self,
        rated_capacity_c: float,
        rated_input_power_c: float,
        rated_capacity_h: float,
        rated_input_power_h: float,
    ) -> None:
        """Initialize VRF (ESS) with rated capacities and power inputs.

        Args:
            rated_capacity_c: Rated cooling capacity [kW].
            rated_input_power_c: Rated cooling power input [kW].
            rated_capacity_h: Rated heating capacity [kW].
            rated_input_power_h: Rated heating power input [kW].
        """
        self.rated_capacity_c = rated_capacity_c
        self.rated_input_power_c = rated_input_power_c
        self.rated_capacity_h = rated_capacity_h
        self.rated_input_power_h = rated_input_power_h
        self.capacity_c = 0.0
        self.input_power_c = 0.0
        self.cop_c = 0.0
        self.capacity_h = 0.0
        self.input_power_h = 0.0
        self.cop_h = 0.0

    def cal_c(self, odb: float, indoor_capacity: float) -> tuple[float, float, float]:
        """Calculate cooling performance.

        Args:
            odb: Outdoor dry-bulb temperature [°C].
            indoor_capacity: Sum of indoor unit rated capacities [kW].

        Returns:
            Tuple of ``(capacity_c, input_power_c, cop_c)``.
        """
        odb = max(15.0, min(43.0, odb))
        k_1 = -0.0025 * odb + 1.0875
        capacity_a = self.rated_capacity_c * k_1
        self.capacity_c = capacity_a

        cr = indoor_capacity / self.rated_capacity_c
        if cr < 1:
            self.capacity_c = indoor_capacity
            if capacity_a < indoor_capacity:
                self.capacity_c = capacity_a

        plr = self.capacity_c / capacity_a
        plr = max(0.3, min(1.0, plr))

        k_2 = 0.0001212 * odb ** 2 + 0.00369 * odb + 0.72238
        k_3 = 0.8573 * plr ** 2 - 0.0456 * plr + 0.1883

        self.input_power_c = self.rated_input_power_c * k_2 * k_3
        self.cop_c = self.capacity_c / self.input_power_c

        return self.capacity_c, self.input_power_c, self.cop_c

    def cal_h(self, owb: float, indoor_capacity: float) -> tuple[float, float, float]:
        """Calculate heating performance.

        Args:
            owb: Outdoor wet-bulb temperature [°C].
            indoor_capacity: Sum of indoor unit rated capacities [kW].

        Returns:
            Tuple of ``(capacity_h, input_power_h, cop_h)``.
        """
        owb = max(-20.0, min(15.0, owb))

        k_1 = 0.0
        if -20 <= owb <= -8:
            k_1 = 0.0255 * owb + 0.847
        elif -8 < owb <= 4.5:
            k_1 = 0.0153 * owb + 0.762
        elif 4.5 < owb <= 15:
            k_1 = 0.0255 * owb + 0.847
        capacity_a = self.rated_capacity_h * k_1
        self.capacity_h = capacity_a

        cr = indoor_capacity / self.rated_capacity_h
        if cr < 1:
            self.capacity_h = indoor_capacity
            if capacity_a < indoor_capacity:
                self.capacity_h = capacity_a

        plr = self.capacity_h / capacity_a
        plr = max(0.3, min(1.0, plr))

        k_2 = 0.0128 * owb + 0.9232
        k_3 = 0.7823 * plr ** 2 + 0.0398 * plr + 0.1779
        self.input_power_h = self.rated_input_power_h * k_2 * k_3
        self.cop_h = self.capacity_h / self.input_power_h

        return self.capacity_h, self.input_power_h, self.cop_h


# ---------------------------------------------------------------------------
# VariableRefrigerantFlowEP  (cooling mode)
# ---------------------------------------------------------------------------

class VariableRefrigerantFlowEP:
    """VRF system (cooling mode) modeled per EnergyPlus system-curve method.

    Regression coefficients are derived from data in ``equipment_spec.xlsx``.

    References:
        EnergyPlus Engineering Reference (22.1), System Curve based VRF Model.
        Raustad, R.A. (2012), Creating Performance Curves for VRF in EnergyPlus.

    Attributes:
        rated_capacity: Rated cooling capacity [kW].
        rated_input_power: Rated cooling power input [kW].
        plr_min: Minimum operating part-load ratio.
        length: Equivalent piping length [m].
        height: Vertical height difference between highest/lowest indoor units [m].
        cr: Combination ratio (indoor / outdoor rated capacity).
    """

    def __init__(
        self,
        rated_capacity: float = 31.6548,
        rated_input_power: float = 9.73,
        length: float = 10.0,
        height: float = 5.0,
    ) -> None:
        """Initialize VRF-EP cooling model and load regression datasets.

        Args:
            rated_capacity: Rated cooling capacity [kW].
            rated_input_power: Rated cooling power input [kW].
            length: Equivalent piping length [m].
            height: Vertical height difference [m].
        """
        self.rated_capacity = rated_capacity
        self.rated_input_power = rated_input_power
        self.plr_min = 0.2
        self.length = length
        self.height = height
        self.indoor_capacity = 0.0
        self.cr = 0.0

        boundary = pd.read_excel("equipment_spec.xlsx", sheet_name="boundary_dataset", header=None)
        boundary = boundary.drop(boundary.index[0])
        self.boundary = pd.DataFrame(boundary, dtype="float64")

        low_temp_c = pd.read_excel("equipment_spec.xlsx", sheet_name="lowt_dataset_c", header=None)
        low_temp_c = low_temp_c.drop(low_temp_c.index[0])
        self.low_temp_c = pd.DataFrame(low_temp_c, dtype="float64")

        low_temp_p = pd.read_excel("equipment_spec.xlsx", sheet_name="lowt_dataset_p", header=None)
        low_temp_p = low_temp_p.drop(low_temp_p.index[0])
        self.low_temp_p = pd.DataFrame(low_temp_p, dtype="float64")

        high_temp_c = pd.read_excel("equipment_spec.xlsx", sheet_name="hight_dataset_c", header=None)
        high_temp_c = high_temp_c.drop(high_temp_c.index[0])
        self.high_temp_c = pd.DataFrame(high_temp_c, dtype="float64")

        high_temp_p = pd.read_excel("equipment_spec.xlsx", sheet_name="hight_dataset_p", header=None)
        high_temp_p = high_temp_p.drop(high_temp_p.index[0])
        self.high_temp_p = pd.DataFrame(high_temp_p, dtype="float64")

    def get_cr_correction(self) -> float:
        """Return the combination-ratio correction factor (≥ 1).

        Returns:
            Correction factor applied when ``cr > 1``.
        """
        cr_data = pd.read_excel("equipment_spec.xlsx", sheet_name="cr_correction", header=None)
        cr_data = cr_data.drop(cr_data.index[0])
        cr_data = pd.DataFrame(cr_data, dtype="float64")

        x_data = np.array(cr_data.iloc[:, 1]).reshape(-1, 1)
        y_data = np.array(cr_data.iloc[:, 0]).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_data, y_data)
        cr_correction_factor = float(model.coef_ * self.cr + model.intercept_)
        return max(1.0, cr_correction_factor)

    def get_eirfplr(self) -> float:
        """Return the energy input ratio modifier as a function of PLR.

        Returns:
            EIRfPLR value for the current combination ratio.
        """
        eirfplr_data = pd.read_excel("equipment_spec.xlsx", sheet_name="eirfplr", header=None)
        eirfplr_data = eirfplr_data.drop(eirfplr_data.index[0])
        eirfplr_data = eirfplr_data.dropna(how="all", axis=1)
        eirfplr_data = pd.DataFrame(eirfplr_data, dtype="float64")

        x_data = eirfplr_data.iloc[:, 3:]
        y_data = eirfplr_data.iloc[:, 1]
        model = LinearRegression()
        model.fit(x_data, y_data)
        a, b, c = model.coef_
        d = model.intercept_
        return a * self.cr + b * self.cr ** 2 + c * self.cr ** 3 + d

    def get_piping_correction(self) -> tuple[float, float]:
        """Return piping correction factors for length and height.

        Returns:
            Tuple of ``(correction_length, correction_height)``.
        """
        pipe_data = pd.read_excel("equipment_spec.xlsx", sheet_name="piping_correction", header=None)
        pipe_data = pipe_data.drop(pipe_data.index[0])
        pipe_data = pd.DataFrame(pipe_data, dtype="float64")

        x_data = pipe_data.iloc[:, 1:]
        y_data = pipe_data.iloc[:, 0]
        model = LinearRegression()
        model.fit(x_data, y_data)
        a = model.intercept_
        b, c, d, e, f = model.coef_
        piping_correction_length = (
            a + b * self.length + c * self.length ** 2
            + d * self.cr + e * self.cr ** 2 + f * self.length * self.cr
        )
        piping_correction_height = 1 - 0.0019231 * self.height
        return piping_correction_length, piping_correction_height

    def cal_loss(
        self, iwb: float, odb: float, indoor_capacity: float
    ) -> tuple[float, float, float]:
        """Calculate performance with piping losses.

        Args:
            iwb: Indoor wet-bulb temperature [°C].
            odb: Outdoor dry-bulb temperature [°C].
            indoor_capacity: Sum of indoor unit capacities [kW].

        Returns:
            Tuple of ``(capacity, input_power, cop)``.
        """
        self.indoor_capacity = indoor_capacity
        self.cr = min(1.5, self.indoor_capacity / self.rated_capacity)
        capacity, input_power, cop = self.cal(iwb, odb)
        pl, ph = self.get_piping_correction()
        capacity_a = capacity * pl * ph

        if self.cr == 1:
            cop = capacity_a / input_power
            return capacity_a, input_power, cop

        cr_correction_factor = self.get_cr_correction()
        eirfplr = self.get_eirfplr()
        plr = self.indoor_capacity / capacity_a
        cr_limit = self.plr_min * capacity_a / self.rated_capacity

        if self.cr > 1:
            capacity_h = capacity_a * cr_correction_factor
            cop = capacity_h / input_power
            return capacity_h, input_power, cop

        if cr_limit <= self.cr < 1:
            input_power_l = input_power * eirfplr
            capacity_l = min(self.indoor_capacity, capacity_a)
            cop = capacity_l / input_power_l
            return capacity_l, input_power_l, cop

        if 0 < self.cr < cr_limit:
            cycling_ratio = plr / self.plr_min
            cycling_ratio_fraction = 0.15 * cycling_ratio + 0.85
            rtf = cycling_ratio / cycling_ratio_fraction
            input_power_min = input_power * eirfplr * rtf
            cop = self.indoor_capacity / input_power_min
            return self.indoor_capacity, input_power_min, cop

    def cal_pl(
        self, iwb: float, odb: float, indoor_capacity: float
    ) -> tuple[float, float, float]:
        """Calculate performance with combination-ratio correction (no piping loss).

        Args:
            iwb: Indoor wet-bulb temperature [°C].
            odb: Outdoor dry-bulb temperature [°C].
            indoor_capacity: Sum of indoor unit capacities [kW].

        Returns:
            Tuple of ``(capacity, input_power, cop)``.
        """
        self.indoor_capacity = indoor_capacity
        self.cr = min(1.5, self.indoor_capacity / self.rated_capacity)
        capacity, input_power, cop = self.cal(iwb, odb)
        cr_correction_factor = self.get_cr_correction()
        eirfplr = self.get_eirfplr()
        plr = self.indoor_capacity / capacity
        cr_limit = self.plr_min * capacity / self.rated_capacity

        if self.cr >= 1:
            capacity_h = capacity * cr_correction_factor
            cop = capacity_h / input_power
            return capacity_h, input_power, cop

        if cr_limit <= self.cr < 1:
            input_power_l = input_power * eirfplr
            capacity_l = min(self.indoor_capacity, capacity)
            cop = capacity_l / input_power_l
            return capacity_l, input_power_l, cop

        if 0 < self.cr < cr_limit:
            cycling_ratio = plr / self.plr_min
            cycling_ratio_fraction = 0.15 * cycling_ratio + 0.85
            rtf = cycling_ratio / cycling_ratio_fraction
            input_power_min = input_power * eirfplr * rtf
            cop = self.indoor_capacity / input_power_min
            return self.indoor_capacity, input_power_min, cop

    def cal(self, iwb: float, odb: float) -> tuple[float, float, float]:
        """Calculate base performance at ``cr = 1`` (no part-load correction).

        Uses low- or high-temperature performance curves depending on *odb*
        relative to the boundary temperature.

        Args:
            iwb: Indoor wet-bulb temperature [°C].
            odb: Outdoor dry-bulb temperature [°C].

        Returns:
            Tuple of ``(capacity, input_power, cop)``.
        """
        x_data_b = self.boundary.iloc[:, 1:]
        y_data_b = self.boundary.iloc[:, 0]
        model_b = LinearRegression()
        model_b.fit(x_data_b, y_data_b)
        odb_boundary = model_b.intercept_ + model_b.coef_[0] * iwb + model_b.coef_[1] * iwb ** 2

        def _fit_capft(df: pd.DataFrame) -> float:
            m = LinearRegression().fit(df.iloc[:, 1:], df.iloc[:, 0])
            ic, co = m.intercept_, m.coef_
            return ic + co[0]*iwb + co[1]*iwb**2 + co[2]*odb + co[3]*odb**2 + co[4]*iwb*odb

        def _fit_eirft(df: pd.DataFrame) -> float:
            m = LinearRegression().fit(df.iloc[:, 3:], df.iloc[:, 2])
            ic, co = m.intercept_, m.coef_
            return ic + co[0]*iwb + co[1]*iwb**2 + co[2]*odb + co[3]*odb**2 + co[4]*iwb*odb

        if odb <= odb_boundary:
            capft = _fit_capft(self.low_temp_c)
            eirft = _fit_eirft(self.low_temp_p)
        else:
            capft = _fit_capft(self.high_temp_c)
            eirft = _fit_eirft(self.high_temp_p)

        capacity = self.rated_capacity * capft
        input_power = self.rated_input_power * eirft * capft
        cop = capacity / input_power
        return capacity, input_power, cop


# ---------------------------------------------------------------------------
# VRFEPHeatingMode
# ---------------------------------------------------------------------------

class VRFEPHeatingMode:
    """VRF system (heating mode) modeled per the EnergyPlus system-curve method.

    Regression coefficients are derived from data in ``equipment_spec.xlsx``.

    Attributes:
        rated_capacity: Rated heating capacity [kW].
        rated_input_power: Rated heating power input [kW].
        plr_min: Minimum operating part-load ratio.
        length: Equivalent piping length [m].
        height: Vertical height difference [m].
        cr: Combination ratio.
    """

    def __init__(
        self,
        rated_capacity: float = 37.5,
        rated_input_power: float = 10.59,
        length: float = 10.0,
        height: float = 5.0,
    ) -> None:
        """Initialize VRF-EP heating model and load regression datasets.

        Args:
            rated_capacity: Rated heating capacity [kW].
            rated_input_power: Rated heating power input [kW].
            length: Equivalent piping length [m].
            height: Vertical height difference [m].
        """
        self.rated_capacity = rated_capacity
        self.rated_input_power = rated_input_power
        self.indoor_capacity = 0.0
        self.cr = 0.0
        self.plr_min = 0.2
        self.length = length
        self.height = height

        def _load(sheet: str) -> pd.DataFrame:
            df = pd.read_excel("equipment_spec.xlsx", sheet_name=sheet, header=None)
            df = df.drop(df.index[0])
            return pd.DataFrame(df, dtype="float64")

        self.boundary_c = _load("boundary_dataset_c")
        self.boundary_p = _load("boundary_dataset_p")
        self.low_temp_c = _load("lowt_dataset_c_h")
        self.low_temp_p = _load("lowt_dataset_p_h")
        self.high_temp_c = _load("hight_dataset_c_h")
        self.high_temp_p = _load("hight_dataset_p_h")

    def get_cr_correction(self) -> float:
        """Return the combination-ratio correction factor for heating mode.

        Returns:
            Correction factor (≥ 1).
        """
        cr_data = pd.read_excel("equipment_spec.xlsx", sheet_name="cr_correction_h", header=None)
        cr_data = cr_data.drop(cr_data.index[0])
        cr_data = pd.DataFrame(cr_data, dtype="float64")

        model = LinearRegression().fit(cr_data.iloc[:, 1:], cr_data.iloc[:, 0])
        a = model.intercept_
        b, c, d = model.coef_
        cr_correction_factor = a + b * self.cr + c * self.cr ** 2 + d * self.cr ** 3
        return max(1.0, cr_correction_factor)

    def get_eirfplr(self) -> float:
        """Return EIRfPLR for the current combination ratio (heating mode).

        Returns:
            EIRfPLR value.
        """
        if self.cr <= 1:
            sheet = "eirfplr_l"
        else:
            sheet = "eirfplr_h"

        df = pd.read_excel("equipment_spec.xlsx", sheet_name=sheet, header=None)
        df = df.drop(df.index[0]).dropna(how="all", axis=1)
        df = pd.DataFrame(df, dtype="float64")

        model = LinearRegression().fit(df.iloc[:, 3:], df.iloc[:, 1])
        if self.cr <= 1:
            a, b, c = model.coef_
            d = model.intercept_
        else:
            a, b, c = model.coef_
            d = model.intercept_
        return a + b * self.cr + c * self.cr ** 2 + d * self.cr ** 3

    def get_piping_correction(self) -> float:
        """Return combined piping correction factor for heating mode.

        Returns:
            Piping correction factor (length + height components summed).
        """
        pipe_data = pd.read_excel("equipment_spec.xlsx", sheet_name="piping_correction_h", header=None)
        pipe_data = pipe_data.drop(pipe_data.index[0])
        pipe_data = pd.DataFrame(pipe_data, dtype="float64")

        model = LinearRegression().fit(pipe_data.iloc[:, 1:], pipe_data.iloc[:, 0])
        a = model.intercept_
        b, c, d = model.coef_
        correction_length = a + b * self.length + c * self.length ** 2 + d * self.length ** 3
        return correction_length + 0.0  # height correction is 0 in heating mode

    def get_defrost_correction(self, owb: float) -> float:
        """Return the defrost correction factor.

        Args:
            owb: Outdoor wet-bulb temperature [°C]; clamped to [-10, 5.84].

        Returns:
            Defrost correction factor.
        """
        owb = max(-10.0, min(5.84, owb))
        df = pd.read_excel("equipment_spec.xlsx", sheet_name="df_correction", header=None)
        df = df.drop(df.index[0])
        df = pd.DataFrame(df, dtype="float64")

        model = LinearRegression().fit(df.iloc[:, 1:], df.iloc[:, 0])
        a = model.intercept_
        b, c, d = model.coef_
        return a + b * owb + c * owb ** 2 + d * owb ** 3

    def cal_loss(
        self, idb: float, owb: float, indoor_capacity: float
    ) -> tuple[float, float, float]:
        """Calculate heating performance with piping and defrost corrections.

        Args:
            idb: Indoor dry-bulb temperature [°C].
            owb: Outdoor wet-bulb temperature [°C].
            indoor_capacity: Sum of indoor unit capacities [kW].

        Returns:
            Tuple of ``(capacity, input_power, cop)``.
        """
        self.indoor_capacity = indoor_capacity
        self.cr = min(1.5, self.indoor_capacity / self.rated_capacity)
        piping_correction = self.get_piping_correction()
        defrost_correction = self.get_defrost_correction(owb=owb)

        capacity, input_power, cop = self.cal(idb, owb)
        capacity_a = capacity * defrost_correction * piping_correction

        if self.cr == 1:
            cop = capacity_a / input_power
            return capacity_a, input_power, cop

        cr_correction_factor = self.get_cr_correction()
        eirfplr = self.get_eirfplr()
        plr = self.indoor_capacity / capacity_a
        cr_limit = self.plr_min * capacity_a / self.rated_capacity

        if self.cr > 1:
            capacity_h = capacity_a * cr_correction_factor
            cop = capacity_h / input_power
            return capacity_h, input_power, cop

        if cr_limit <= self.cr < 1:
            input_power_l = input_power * eirfplr
            capacity_l = min(self.indoor_capacity, capacity_a)
            cop = capacity_l / input_power_l
            return capacity_l, input_power_l, cop

        if 0 < self.cr < cr_limit:
            cycling_ratio = plr / self.plr_min
            cycling_ratio_fraction = 0.15 * cycling_ratio + 0.85
            rtf = cycling_ratio / cycling_ratio_fraction
            input_power_min = input_power * eirfplr * rtf
            cop = self.indoor_capacity / input_power_min
            return self.indoor_capacity, input_power_min, cop

    def cal_pl(
        self, idb: float, owb: float, indoor_capacity: float
    ) -> tuple[float, float, float]:
        """Calculate heating performance with combination-ratio correction only.

        Args:
            idb: Indoor dry-bulb temperature [°C].
            owb: Outdoor wet-bulb temperature [°C].
            indoor_capacity: Sum of indoor unit capacities [kW].

        Returns:
            Tuple of ``(capacity, input_power, cop)``.
        """
        self.indoor_capacity = indoor_capacity
        self.cr = min(1.5, self.indoor_capacity / self.rated_capacity)
        capacity, input_power, cop = self.cal(idb, owb)
        cr_correction_factor = self.get_cr_correction()
        eirfplr = self.get_eirfplr()
        plr = self.indoor_capacity / capacity
        cr_limit = self.plr_min * capacity / self.rated_capacity

        if self.cr >= 1:
            capacity_h = capacity * cr_correction_factor
            cop = capacity_h / input_power
            return capacity_h, input_power, cop

        if cr_limit <= self.cr < 1:
            input_power_l = input_power * eirfplr
            capacity_l = min(self.indoor_capacity, capacity)
            cop = capacity_l / input_power_l
            return capacity_l, input_power_l, cop

        if 0 < self.cr < cr_limit:
            cycling_ratio = plr / self.plr_min
            cycling_ratio_fraction = 0.15 * cycling_ratio + 0.85
            rtf = cycling_ratio / cycling_ratio_fraction
            input_power_min = input_power * eirfplr * rtf
            cop = self.indoor_capacity / input_power_min
            return self.indoor_capacity, input_power_min, cop

    def cal(self, idb: float, owb: float) -> tuple[float, float, float]:
        """Calculate base heating performance at ``cr = 1``.

        Selects low- or high-temperature curves based on *owb* vs the
        boundary temperature computed separately for capacity and power.

        Args:
            idb: Indoor dry-bulb temperature [°C].
            owb: Outdoor wet-bulb temperature [°C].

        Returns:
            Tuple of ``(capacity, input_power, cop)``.
        """
        def _boundary(df: pd.DataFrame) -> float:
            m = LinearRegression().fit(df.iloc[:, 1:], df.iloc[:, 0])
            return m.intercept_ + m.coef_[0] * idb + m.coef_[1] * idb ** 2

        def _fit_capft(df: pd.DataFrame) -> float:
            m = LinearRegression().fit(df.iloc[:, 1:], df.iloc[:, 0])
            ic, co = m.intercept_, m.coef_
            return ic + co[0]*idb + co[1]*idb**2 + co[2]*owb + co[3]*owb**2 + co[4]*idb*owb

        def _fit_eirft(df: pd.DataFrame) -> float:
            m = LinearRegression().fit(df.iloc[:, 3:], df.iloc[:, 2])
            ic, co = m.intercept_, m.coef_
            return ic + co[0]*idb + co[1]*idb**2 + co[2]*owb + co[3]*owb**2 + co[4]*idb*owb

        boundary_c = _boundary(self.boundary_c)
        boundary_p = _boundary(self.boundary_p)

        capft_l = _fit_capft(self.low_temp_c)
        capft_h = _fit_capft(self.high_temp_c)
        eirft_l = _fit_eirft(self.low_temp_p)
        eirft_h = _fit_eirft(self.high_temp_p)

        if owb <= boundary_c and owb <= boundary_p:
            capft, eirft = capft_l, eirft_l
        elif owb > boundary_c and owb > boundary_p:
            capft, eirft = capft_h, eirft_h
        elif boundary_c > owb > boundary_p:
            capft, eirft = capft_l, eirft_h
        else:
            capft, eirft = capft_h, eirft_l

        capacity = self.rated_capacity * capft
        input_power = self.rated_input_power * eirft * capft
        cop = capacity / input_power
        return capacity, input_power, cop


# ---------------------------------------------------------------------------
# GeoThermalHeatPump_LCEM
# ---------------------------------------------------------------------------

class GeoThermalHeatPump_LCEM:
    """Ground-source heat pump modeled using the LCEM tool equations.

    Performance is described by empirical polynomial equations for cooling and
    heating capacity, full-load power, and part-load power ratio.

    Attributes:
        temp_c: Chilled-water outlet temperature setpoint [°C].
        temp_h: Hot-water outlet temperature setpoint [°C].
        rated_cap_c: Rated cooling capacity [kW].
        rated_cap_h: Rated heating capacity [kW].
        rated_power_c: Rated cooling power [kW].
        rated_power_h: Rated heating power [kW].
        rated_freq: Rated compressor frequency [Hz].
    """

    def __init__(
        self,
        temp_c: float = 7.0,
        temp_h: float = 45.0,
        rank_c: int = 1,
        rank_h: int = 1,
        rated_cap_c: float = 53.0,
        rated_cap_h: float = 61.0,
        rated_flow_c: float = 153.0,
        rated_flow_h: float = 167.0,
        rated_flow_cd_c: float = 183.0,
        rated_flow_cd_h: float = 133.0,
        rated_power_c: float = 10.8,
        rated_power_h: float = 15.2,
        rated_freq: float = 50.0,
        coefficient_ele_a: float = 1.0,
        coefficient_ele_b: float = 0.0,
        mod_temp_chs: float = 0.01,
    ) -> None:
        """Initialize ground-source heat pump with rated parameters.

        Args:
            temp_c: Chilled-water outlet setpoint [°C].
            temp_h: Hot-water outlet setpoint [°C].
            rank_c: Cooling rank (number of units).
            rank_h: Heating rank (number of units).
            rated_cap_c: Rated cooling capacity [kW].
            rated_cap_h: Rated heating capacity [kW].
            rated_flow_c: Rated chilled-water flow rate [L/min].
            rated_flow_h: Rated hot-water flow rate [L/min].
            rated_flow_cd_c: Rated heat-source water flow rate (cooling) [L/min].
            rated_flow_cd_h: Rated heat-source water flow rate (heating) [L/min].
            rated_power_c: Rated cooling power [kW].
            rated_power_h: Rated heating power [kW].
            rated_freq: Rated compressor frequency [Hz].
            coefficient_ele_a: Linear coefficient for power correction.
            coefficient_ele_b: Constant for power correction.
            mod_temp_chs: Temperature step for outlet-temperature iteration [°C].
        """
        self.temp_c = temp_c
        self.temp_h = temp_h
        self.rank_c = rank_c
        self.rank_h = rank_h
        self.rated_cap_c = rated_cap_c
        self.rated_cap_h = rated_cap_h
        self.rated_flow_c = rated_flow_c
        self.rated_flow_h = rated_flow_h
        self.rated_flow_cd_c = rated_flow_cd_c
        self.rated_flow_cd_h = rated_flow_cd_h
        self.rated_power_c = rated_power_c
        self.rated_power_h = rated_power_h
        self.rated_freq = rated_freq
        self.coefficient_ele_a = coefficient_ele_a
        self.coefficient_ele_b = coefficient_ele_b
        self.mod_temp_chs = mod_temp_chs

    def get_config(self) -> dict:
        """Return current configuration parameters as a dictionary.

        Returns:
            Dict mapping parameter names to current values.
        """
        return {
            "temp_c": self.temp_c,
            "temp_h": self.temp_h,
            "rank_c": self.rank_c,
            "rank_h": self.rank_h,
            "rated_cap_c": self.rated_cap_c,
            "rated_cap_h": self.rated_cap_h,
            "rated_flow_c": self.rated_flow_c,
            "rated_flow_h": self.rated_flow_h,
            "rated_flow_cd_c": self.rated_flow_cd_c,
            "rated_flow_cd_h": self.rated_flow_cd_h,
            "rated_power_c": self.rated_power_c,
            "rated_power_h": self.rated_power_h,
            "rated_freq": self.rated_freq,
            "coefficient_ele_a": self.coefficient_ele_a,
            "coefficient_ele_b": self.coefficient_ele_b,
            "mod_temp_chs": self.mod_temp_chs,
        }

    def set_config(
        self,
        temp_c: float | None = None,
        temp_h: float | None = None,
        rank_c: int | None = None,
        rank_h: int | None = None,
        rated_cap_c: float | None = None,
        rated_cap_h: float | None = None,
        rated_flow_c: float | None = None,
        rated_flow_h: float | None = None,
        rated_flow_cd_c: float | None = None,
        rated_flow_cd_h: float | None = None,
        rated_power_c: float | None = None,
        rated_power_h: float | None = None,
        rated_freq: float | None = None,
        coefficient_ele_a: float | None = None,
        coefficient_ele_b: float | None = None,
        mod_temp_chs: float | None = None,
    ) -> None:
        """Update configuration parameters; ``None`` values are left unchanged.

        Args:
            temp_c: Chilled-water outlet setpoint [°C].
            temp_h: Hot-water outlet setpoint [°C].
            rank_c: Cooling rank.
            rank_h: Heating rank.
            rated_cap_c: Rated cooling capacity [kW].
            rated_cap_h: Rated heating capacity [kW].
            rated_flow_c: Rated chilled-water flow [L/min].
            rated_flow_h: Rated hot-water flow [L/min].
            rated_flow_cd_c: Rated heat-source flow (cooling) [L/min].
            rated_flow_cd_h: Rated heat-source flow (heating) [L/min].
            rated_power_c: Rated cooling power [kW].
            rated_power_h: Rated heating power [kW].
            rated_freq: Rated compressor frequency [Hz].
            coefficient_ele_a: Linear power correction coefficient.
            coefficient_ele_b: Constant power correction term.
            mod_temp_chs: Outlet-temperature iteration step [°C].
        """
        if temp_c is not None:
            self.temp_c = temp_c
        if temp_h is not None:
            self.temp_h = temp_h
        if rank_c is not None:
            self.rank_c = rank_c
        if rank_h is not None:
            self.rank_h = rank_h
        if rated_cap_c is not None:
            self.rated_cap_c = rated_cap_c
        if rated_cap_h is not None:
            self.rated_cap_h = rated_cap_h
        if rated_flow_c is not None:
            self.rated_flow_c = rated_flow_c
        if rated_flow_h is not None:
            self.rated_flow_h = rated_flow_h
        if rated_flow_cd_c is not None:
            self.rated_flow_cd_c = rated_flow_cd_c
        if rated_flow_cd_h is not None:
            self.rated_flow_cd_h = rated_flow_cd_h
        if rated_power_c is not None:
            self.rated_power_c = rated_power_c
        if rated_power_h is not None:
            self.rated_power_h = rated_power_h
        if rated_freq is not None:            # Bug fix: was bare `rated_freq`
            self.rated_freq = rated_freq
        if coefficient_ele_a is not None:
            self.coefficient_ele_a = coefficient_ele_a
        if coefficient_ele_b is not None:
            self.coefficient_ele_b = coefficient_ele_b
        if mod_temp_chs is not None:
            self.mod_temp_chs = mod_temp_chs

    def run(
        self,
        state: int,
        mode: int,
        flow_ch: float,
        temp_chr: float,
        flow_cd: float,
        temp_cds: float,
    ) -> dict:
        """Run the GSHP model for one time step.

        Args:
            state: Operation flag (0 = stop, 1 = run).
            mode: Mode flag (0 = stop, 1 = cooling, 2 = heating).
            flow_ch: Chilled/hot-water flow rate [L/min].
            temp_chr: Chilled/hot-water return temperature [°C].
            flow_cd: Heat-source water flow rate [L/min].
            temp_cds: Heat-source water supply temperature [°C].

        Returns:
            Dict with keys: ``temp_chs``, ``temp_cdr``, ``flow_cd``,
            ``cooling``, ``heating``, ``PLR``, ``COP``, ``power``, ``gas``,
            ``error``.
        """
        error = 0
        tmp_temp_chs = float(self.temp_h) if mode == 2 else float(self.temp_c)

        if state * mode == 0:
            work_ch = 0.0
            upper_work_ch = 0.0
            full_power = 0.0
            full_cop = 0.0
            plr = 0.0
            mod_plr = 0.0
        else:
            work_ch = self.rated_cap_c if mode == 1 else self.rated_cap_h
            upper_work_ch = 0.0

            while work_ch > upper_work_ch * 1.05:
                self.upper_work_ch = upper_work_ch
                self.temp_chs = tmp_temp_chs
                self.work_ch = work_ch

                work_ch = abs(tmp_temp_chs - temp_chr) * flow_ch * 60 / 860
                if mode == 1:
                    if self.rated_freq == 50.0:
                        upper_work_ch = (
                            0.02251*tmp_temp_chs**2 + (-0.01407)*tmp_temp_chs*temp_cds
                            + (-0.00126)*temp_cds**2 + 1.738*tmp_temp_chs
                            + (-0.305)*temp_cds + 51.32
                        )
                        full_power = (
                            (-0.001418)*tmp_temp_chs**2 + 0.005793*tmp_temp_chs*temp_cds
                            + (-0.0001034)*temp_cds**2 + (-0.1244)*tmp_temp_chs
                            + 0.2912*temp_cds + 3.218
                        )
                    else:
                        upper_work_ch = (
                            0.02386*tmp_temp_chs**2 + (-0.0165)*tmp_temp_chs*temp_cds
                            + (-0.001652)*temp_cds**2 + 1.997*tmp_temp_chs
                            + (-0.3499)*temp_cds + 60.22
                        )
                        full_power = (
                            (-0.0006445)*tmp_temp_chs**2 + 0.006063*tmp_temp_chs*temp_cds
                            + 0.0004485*temp_cds**2 + (-0.1096)*tmp_temp_chs
                            + 0.319*temp_cds + 4.412
                        )
                    upper_work_ch = upper_work_ch / 53.0 * self.rated_cap_c
                    full_power = full_power / 10.8 * self.rated_power_c
                else:
                    if self.rated_freq == 50.0:
                        upper_work_ch = (
                            (-0.001352)*tmp_temp_chs**2 + (-0.008325)*tmp_temp_chs*temp_cds
                            + 0.02112*temp_cds**2 + 0.03779*tmp_temp_chs
                            + 1.445*temp_cds + 46.62
                        )
                        full_power = (
                            (-0.0001136)*tmp_temp_chs**2 + 0.005796*tmp_temp_chs*temp_cds
                            + (-0.001418)*temp_cds**2 + 0.264*tmp_temp_chs
                            + (-0.1393)*temp_cds + 2.48
                        )
                    else:
                        upper_work_ch = (
                            (-0.001202)*tmp_temp_chs**2 + (-0.01049)*tmp_temp_chs*temp_cds
                            + 0.02322*temp_cds**2 + 0.02972*tmp_temp_chs
                            + 1.709*temp_cds + 55.42
                        )
                        full_power = (
                            0.0004311*tmp_temp_chs**2 + 0.006065*tmp_temp_chs*temp_cds
                            + (-0.0006446)*temp_cds**2 + 0.28542*tmp_temp_chs
                            + (-0.1335)*temp_cds + 3.489
                        )
                    upper_work_ch = upper_work_ch / 61.0 * self.rated_cap_h
                    full_power = full_power / 15.2 * self.rated_power_c
                full_cop = upper_work_ch / full_power

                if work_ch > upper_work_ch * 1.05:
                    if mode == 1:
                        tmp_temp_chs += self.mod_temp_chs
                    else:
                        tmp_temp_chs -= self.mod_temp_chs

            plr = work_ch / upper_work_ch
            mod_plr = max(0.3, plr)

        if state * mode == 0:
            rate_power = 0.0
            power = 0.0
            cop = 0.0
            temp_cdr = temp_cds
            work_c = work_ch if mode == 1 else 0.0
            work_h = 0.0 if mode == 1 else work_ch
        else:
            if mode == 1:
                if self.rated_freq == 50.0:
                    rate_power = (
                        4.899*mod_plr**4 + (-12.81)*mod_plr**3
                        + 12.05*mod_plr**2 + (-3.792)*mod_plr + 0.6557
                    )
                else:
                    rate_power = (
                        (-1.304)*mod_plr**4 + 4.172*mod_plr**3
                        + (-4.376)*mod_plr**2 + 2.747*mod_plr + (-0.2371)
                    )
            else:
                if self.rated_freq == 50.0:
                    rate_power = (
                        1.804*mod_plr**4 + (-4.428)*mod_plr**3
                        + 3.858*mod_plr**2 + (-0.4286)*mod_plr + 0.1962
                    )
                else:
                    rate_power = (
                        (-0.3699)*mod_plr**4 + 1.255*mod_plr**3
                        + (-1.309)*mod_plr**2 + 1.477*mod_plr + (-0.04922)
                    )

            if work_ch == 0.0:
                power = 0.0
                cop = 0.0
            else:
                power = max(0.0, self.coefficient_ele_a * full_power * rate_power + self.coefficient_ele_b)
                cop = work_ch / power

            if flow_ch == 0:
                temp_cdr = temp_cds
            elif mode == 1:
                temp_cdr = temp_cds + (work_ch + power) * 860 / 60 / flow_cd
            else:
                temp_cdr = temp_cds + (-work_ch + power) * 860 / 60 / flow_cd

            work_c = work_ch if mode == 1 else 0.0
            work_h = 0.0 if mode == 1 else work_ch

        return {
            "temp_chs": tmp_temp_chs,
            "temp_cdr": temp_cdr,
            "flow_cd": flow_cd,
            "cooling": work_c,
            "heating": work_h,
            "PLR": mod_plr,
            "COP": cop,
            "power": power,
            "gas": 0.0,
            "error": error,
        }


# ---------------------------------------------------------------------------
# CoolingTower
# ---------------------------------------------------------------------------

class CoolingTower:
    """Counter-flow cooling tower using the NTU-effectiveness method.

    UA is calibrated at initialization to match the rated performance point.

    Attributes:
        ua: Calibrated overall heat-transfer coefficient [W/K].
        inv: Fan inverter frequency ratio (0.0–1.0).
        tout_w: Cooling-water outlet temperature [°C].
        pw: Fan power consumption [kW].
        dp: Water-side pressure drop [kPa].
        flag: 0 = converged, 1 = iteration limit reached.
    """

    def __init__(
        self,
        tin_w_d: float = 37.0,
        tout_w_d: float = 32.0,
        twb_d: float = 27.0,
        g_w_d: float = 0.26,
        g_a_d: float = 123.0,
        pw_d: float = 2.4,
        actual_head: float = 2.0,
        kr: float = 1.0,
    ) -> None:
        """Initialize cooling tower and calibrate UA value.

        Args:
            tin_w_d: Rated cooling-water inlet temperature [°C].
            tout_w_d: Rated cooling-water outlet temperature [°C].
            twb_d: Rated outdoor wet-bulb temperature [°C].
            g_w_d: Rated cooling-water flow rate [kg/s].
            g_a_d: Rated air flow rate [m³/min].
            pw_d: Rated fan power [kW].
            actual_head: Static head of water [m].
            kr: Pressure-loss coefficient [kPa/(m³/min)²].
        """
        self.kr = kr
        self.g_w = 0.0
        self.tin_w = 15.0
        self.tou_w = 15.0
        self.g_a = 0.0
        self.tdb = 15.0
        self.rh = 50.0
        self.inv = 1.0
        self.flag = 0
        self.pw = 0.0
        self.dp = 0.0
        self.tout_w = 15.0
        self.actual_head = actual_head
        self.g_a_d = g_a_d
        self.pw_d = pw_d
        self.tin_w_d = tin_w_d
        self.tout_w_d = tout_w_d
        self.twb_d = twb_d
        self.g_w_d = g_w_d

        # Calibrate UA
        ua_min = 0.1
        ua_max = 9999999999.0
        tout_w0 = self.tout_w_d + 1.0
        cnt = 0
        while abs(tout_w0 - self.tout_w_d) > 0.001:
            self.ua = (ua_min + ua_max) / 2
            tout_w0 = self.cal(self.g_w_d, self.tin_w_d, self.twb_d, 100)
            if tout_w0 - self.tout_w_d > 0:
                ua_min = self.ua
            else:
                ua_max = self.ua
            cnt += 1
            if cnt == 100:
                print("The ua value for cooling tower is not calibrated appropriately")
                break

        self.inv = 0.0

    def cal(self, g_w: float, tin_w: float, tdb: float, rh: float) -> float:
        """Calculate cooling-water outlet temperature.

        Args:
            g_w: Cooling-water flow rate [kg/s].
            tin_w: Cooling-water inlet temperature [°C].
            tdb: Outdoor dry-bulb temperature [°C].
            rh: Outdoor relative humidity [%].

        Returns:
            Cooling-water outlet temperature [°C].
        """
        self.g_w = g_w
        self.tdb = tdb
        self.rh = rh
        self.tin_w = tin_w
        self.g_a = self.inv * self.g_a_d
        self.pw = self.pw_d * self.inv ** 3

        if self.g_a < 10:
            self.g_a = 10

        g_w_kg = self.g_w * 10 ** 3 / 60   # m³/min → kg/s
        g_a_kg = self.g_a * 1.293 / 60
        cpw = 4184
        cp = 1100

        if g_w_kg > 0:
            [hin, _] = tdb_rh2h_x(tdb, rh)
            twbin = tdb_rh2twb(tdb, rh)

            twboutmax = max(self.tin_w, twbin)
            twboutmin = min(self.tin_w, twbin)
            twbout0 = 1.0
            twbout = 0.0
            cnt = 0
            q = 0.0
            cw = 0.0
            self.flag = 0

            while (twbout0 - twbout < -0.01) or (twbout0 - twbout > 0.01):
                twbout0 = (twboutmax + twboutmin) / 2
                [hout, _] = tdb_rh2h_x(twbout0, 100)

                dh = (hout - hin) * 1000
                dtwb = twbout0 - twbin
                cpe = dh / dtwb

                ua_e = self.ua * cpe / cp
                cw = g_w_kg * cpw
                ca = g_a_kg * cpe
                cmin = max(0.001, min(cw, ca))
                cmax = max(0.001, max(cw, ca))

                ntu = ua_e / cmin
                eps = (
                    (1 - math.exp(-ntu * (1 - cmin / cmax)))
                    / (1 - cmin / cmax * math.exp(-ntu * (1 - cmin / cmax)))
                )
                q = eps * cmin * (self.tin_w - twbin)
                twbout = twbin + q / ca

                if twbout < twbout0:
                    twboutmax = twbout0
                else:
                    twboutmin = twbout0

                cnt += 1
                if cnt > 30:
                    self.flag = 1
                    break

            self.tout_w = self.tin_w - q / cw

        self.dp = -self.kr * self.g_w ** 2
        return self.tout_w

    def f2p(self, g_w: float) -> float:
        """Calculate pressure drop from cooling-water flow rate.

        Args:
            g_w: Cooling-water flow rate [m³/min].

        Returns:
            Pressure drop [kPa] (negative convention, includes static head).
        """
        self.g_w = g_w
        self.dp = -self.kr * self.g_w ** 2 - 9.8 * self.actual_head
        return self.dp

    def f2p_co(self) -> list[float]:
        """Return polynomial coefficients for f2p.

        Returns:
            Coefficients ``[c0, c1, c2]`` where dp = c0 + c1*g + c2*g².
        """
        return [-9.8 * self.actual_head, 0, -self.kr]


# ---------------------------------------------------------------------------
# AHU_simple
# ---------------------------------------------------------------------------

class AHU_simple:
    """Simplified air-handling unit with fixed maximum supply temperature.

    The outlet temperature is clipped at 25 °C (``flag = 1`` when clipped).

    Attributes:
        kr: Pressure-loss coefficient [kPa/(m³/min)²].
        g: Flow rate [m³/min].
        dp: Pressure drop [kPa].
        tin: Water inlet temperature [°C].
        tout: Water outlet temperature [°C].
        q_load: Load heat [kJ/min] (input).
        q: Processed heat [kJ/min].
        flag: 0 = normal, 1 = outlet capped at 25 °C.
    """

    def __init__(self, kr: float = 1.0) -> None:
        """Initialize AHU with pressure-loss coefficient.

        Args:
            kr: Pressure-loss coefficient [kPa/(m³/min)²].
        """
        self.kr = kr
        self.g = 0.0
        self.dp = 0.0
        self.tin = 15.0
        self.tout = 15.0
        self.q_load = 0.0
        self.q = 0.0
        self.flag = 0

    def cal(self, g: float, tin: float, q_load: float) -> float:
        """Calculate AHU outlet temperature.

        Args:
            g: Flow rate [m³/min].
            tin: Water inlet temperature [°C].
            q_load: Load heat [kJ/min].

        Returns:
            Water outlet temperature [°C].
        """
        self.g = g
        self.tin = tin
        self.q_load = q_load
        self.flag = 0
        self.tout = tin + q_load / 4.184 / g if g > 0 else tin

        if self.tout > 25:
            self.tout = 25.0
            self.flag = 1

        self.q = (self.tout - self.tin) * self.g * 4.184
        self.dp = -self.kr * self.g ** 2
        return self.tout

    def f2p(self, g: float) -> float:
        """Calculate pressure drop from flow rate.

        Args:
            g: Flow rate [m³/min].

        Returns:
            Pressure drop [kPa].
        """
        self.g = g
        self.dp = -self.kr * self.g ** 2
        return self.dp

    def f2p_co(self) -> list[float]:
        """Return polynomial coefficients for f2p.

        Returns:
            Coefficients ``[0, 0, -kr]``.
        """
        return [0, 0, -self.kr]


# ---------------------------------------------------------------------------
# VerticalWaterThermalStorageTank
# ---------------------------------------------------------------------------

class VerticalWaterThermalStorageTank:
    """Thermally stratified vertical water tank (based on the Popolo model).

    Uses a TDMA solver for the 1-D advection–diffusion–decay equation.
    Temperature stratification and buoyancy-driven mixing are modeled.

    Attributes:
        tes_temp: Layer temperature array [°C] (index 0 = top).
        tout: Outlet temperature [°C].
        heat: Stored thermal energy relative to reference temperature [MJ].
        num_layer: Number of computational layers.
        dz: Layer thickness [m].
    """

    # ------------------------------------------------------------------
    # Water property helpers
    # ------------------------------------------------------------------

    def water_density(self, temp: float) -> float:
        """Return water density at *temp* [kg/m³].

        Args:
            temp: Temperature [°C].

        Returns:
            Density [kg/m³].
        """
        return (
            999.83952 + 16.945176*temp - 7.987041e-3*temp**2
            - 46.170461e-6*temp**3 + 105.56302e-9*temp**4 - 280.54253e-12*temp**5
        ) / (1 + 16.879850e-3*temp)

    def water_thermal_conductivity(self, temp: float) -> float:
        """Return water thermal conductivity at *temp* [W/(m·K)].

        Args:
            temp: Temperature [°C].

        Returns:
            Thermal conductivity [W/(m·K)].
        """
        a = [-1.3734399e-1, 4.2128755, -5.9412196, 1.2794890]
        temp_critical_point = 647.096
        tr = (temp_critical_point - (temp + 273.15)) / temp_critical_point
        lamb = a[-1]
        for i in range(len(a) - 2, 0, -1):
            lamb = a[i] + lamb * tr
        return lamb

    def water_specific_heat(self, temp: float) -> float:
        """Return water specific heat at *temp* [kJ/(kg·K)].

        Args:
            temp: Temperature [°C].

        Returns:
            Specific heat [kJ/(kg·K)].
        """
        a = [1.0570130, 2.1952960e1, -4.9895501e1, 3.6963413e1]
        temp_critical_point = 647.096
        tr = (temp_critical_point - (temp + 273.15)) / temp_critical_point
        cpw = a[-1]
        for i in range(len(a) - 2, 0, -1):
            cpw = a[i] + cpw * tr
        return cpw

    def water_thermal_diffusivity(self, temp: float) -> float:
        """Return water thermal diffusivity at *temp* [m²/s].

        Args:
            temp: Temperature [°C].

        Returns:
            Thermal diffusivity [m²/s].
        """
        lamb = self.water_thermal_conductivity(temp)
        cp = self.water_specific_heat(temp)
        rho = self.water_density(temp)
        return lamb / (1000 * cp * rho)

    def TDMA_solver(
        self,
        a: list[float],
        b: list[float],
        c: list[float],
        d: list[float],
    ) -> np.ndarray:
        """Solve a tri-diagonal system using the Thomas algorithm (TDMA).

        Args:
            a: Sub-diagonal coefficients (length n-1).
            b: Main-diagonal coefficients (length n).
            c: Super-diagonal coefficients (length n-1).
            d: Right-hand side vector (length n).

        Returns:
            Solution vector of length n.
        """
        nf = len(d)
        ac, bc, cc, dc = map(np.array, (a, b, c, d))
        for it in range(1, nf):
            mc = ac[it - 1] / bc[it - 1]
            bc[it] -= mc * cc[it - 1]
            dc[it] -= mc * dc[it - 1]
        xc = bc
        xc[-1] = dc[-1] / bc[-1]
        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
        return xc

    def __init__(
        self,
        timedelta: float,
        depth: float = 6.0,
        base_area: float = 10.0,
    ) -> None:
        """Initialize thermal storage tank geometry and state.

        Args:
            timedelta: Simulation time step [s].
            depth: Tank depth [m].
            base_area: Tank cross-sectional area [m²].
        """
        self.pipeinstal_height = depth - 0.2
        self.d_in = 0.1
        self.depth = depth
        self.base_area = base_area
        self.timedelta = timedelta
        self.tin = 7.0
        self.g_w = 15 / 60
        self.heatloss_coef = 0.001 * 4.8 ** 2 * 6 * 0.04 / 0.4
        self.t_ambient = 25.0
        self.sig_downflow = 0
        self.num_layer = 100

        self.cal_mat = np.zeros((3, self.num_layer), dtype=float)
        self.vec1 = np.zeros(self.num_layer, dtype=float)
        self.vec2 = np.zeros(self.num_layer, dtype=float)

        self.dz = self.depth / self.num_layer
        self.pipeinstal_layer = int(self.pipeinstal_height / self.dz)
        self.t_ref = 15.0
        self.tes_temp = np.full(self.num_layer, self.t_ref, dtype=float)
        self.heat = 0.0
        self.tout = 15.0

    def cal(
        self,
        tin: float,
        g_w: float,
        sig_downflow: int = 0,
        t_ref: float = 15.0,
        t_ambient: float = 25.0,
    ) -> float:
        """Advance the tank state by one time step.

        Args:
            tin: Inlet water temperature [°C].
            g_w: Inlet flow rate [m³/min].
            sig_downflow: 1 = downward flow, 0 = upward flow.
            t_ref: Reference temperature for stored energy calculation [°C].
            t_ambient: Ambient temperature for heat-loss calculation [°C].

        Returns:
            Outlet water temperature [°C].
        """
        self.tin = tin
        self.g_w = g_w
        self.sig_downflow = sig_downflow
        self.t_ambient = t_ambient
        self.t_ref = t_ref
        self.temp_avg = float(np.mean(self.tes_temp))

        if self.g_w == 0:
            self.cal_mat = np.zeros((3, self.num_layer), dtype=float)
            self.vec1 = np.zeros(self.num_layer, dtype=float)
            self.vec2 = np.zeros(self.num_layer, dtype=float)
        elif (self.sig_downflow and self.tin <= self.tes_temp[0]) or (
            not self.sig_downflow and self.tes_temp[self.num_layer - 1] <= self.tin
        ):
            if self.sig_downflow:
                temp_mixavg = self.tes_temp[0]
                temp_mix = self.tes_temp[0] + self.timedelta / (self.base_area * self.dz) * (
                    self.tin - self.tes_temp[0]) * self.g_w / 60
            else:
                temp_mixavg = self.tes_temp[-1]
                temp_mix = self.tes_temp[-1] + self.timedelta / (self.base_area * self.dz) * (
                    self.tin - self.tes_temp[-1])

            for num_mixed in range(1, self.num_layer):
                if self.sig_downflow:
                    if self.tes_temp[num_mixed] < temp_mix:
                        break
                    temp_mix = (temp_mix * num_mixed + self.tes_temp[num_mixed]) / (num_mixed + 1)
                    temp_mixavg += self.tes_temp[num_mixed]
                else:
                    tgtlayer = self.num_layer - (num_mixed + 1)
                    if temp_mixavg < self.tes_temp[tgtlayer]:
                        break
                    temp_mix = 0.5 * (temp_mix + self.tes_temp[tgtlayer])
                    temp_mixavg += self.tes_temp[tgtlayer]

            temp_mixavg /= num_mixed
            bf = self.g_w / 60 / (num_mixed * self.dz)

            for i in range(self.num_layer):
                tgtlayer = i if self.sig_downflow else self.num_layer - (i + 1)
                if i < num_mixed:
                    self.vec1[tgtlayer] = bf
                    self.tes_temp[tgtlayer] = temp_mixavg
                    self.vec2[tgtlayer] = self.dz / self.base_area * self.vec1[tgtlayer]
                else:
                    self.vec1[tgtlayer] = 0.0
                    self.vec2[tgtlayer] = 0.0
                if i != 0:
                    if self.sig_downflow:
                        self.vec2[tgtlayer] += self.vec2[tgtlayer - 1]
                    else:
                        self.vec2[tgtlayer] += self.vec2[tgtlayer + 1]
        else:
            area_pipe = math.pow(self.d_in / 2, 2) * math.pi
            u_waterin2 = math.pow(self.g_w / 60 / area_pipe, 2)
            rho = self.water_density(self.tin)

            tgt = rho * u_waterin2 / (9.8 * self.dz)
            rr = 0.0
            for lmax in range(self.num_layer - 1):
                if self.sig_downflow:
                    rr += self.water_density(self.tes_temp[lmax]) - rho
                else:
                    rr += rho - self.water_density(self.tes_temp[self.num_layer - lmax - 1])
                if rr > tgt:
                    break
            if not self.sig_downflow:
                lmax = self.num_layer - lmax - 1

            temp_lmax = self.tes_temp[lmax]
            if temp_lmax == self.tin:
                lm = self.depth
            else:
                rho0 = self.water_density(temp_lmax)
                self.ar = self.d_in * 9.8 * abs(rho0 - rho) / (rho0 * u_waterin2)
                ndt = 0.0
                if self.sig_downflow:
                    for i in range(lmax):
                        ndt += self.tes_temp[i] - temp_lmax
                else:
                    for i in range(lmax, self.num_layer):
                        ndt += self.tes_temp[i] - temp_lmax
                ndt = (ndt * self.dz) / (self.depth * (self.tin - temp_lmax))
                lm = (self.depth * 0.8 * math.pow(self.ar, -0.5) * self.d_in / self.depth
                      + 0.5 * ndt)

            z1 = 0.0
            bf = self.g_w / 60 / (2 * lm ** 3)
            for i in range(self.num_layer):
                ln = i if self.sig_downflow else self.num_layer - (i + 1)
                z2 = z1 + self.dz
                if z2 < lm:
                    self.vec1[ln] = bf * (3 * lm ** 2 - self.dz ** 2 * (3 * i * (i + 1) + 1))
                elif lm < z1:
                    self.vec1[ln] = 0.0
                else:
                    self.vec1[ln] = bf * math.pow(i * self.dz - lm, 2) * (i + 2 * lm / self.dz)
                if i == 0:
                    self.vec2[ln] = self.dz / self.base_area * self.vec1[ln]
                elif self.sig_downflow:
                    self.vec2[ln] = self.vec2[ln - 1] + self.dz / self.base_area * self.vec1[ln]
                else:
                    self.vec2[ln] = self.vec2[ln + 1] + self.dz / self.base_area * self.vec1[ln]
                z1 = z2

        outlet_layer = self.pipeinstal_layer
        if not self.sig_downflow:
            outlet_layer = self.num_layer - self.pipeinstal_layer - 1

        s = self.timedelta * self.water_thermal_diffusivity(self.temp_avg) / self.dz ** 2
        p = self.heatloss_coef * self.timedelta / (
            self.water_density(self.temp_avg) * 4.186 * self.depth * self.base_area
        )

        for i in range(self.num_layer):
            r1 = 0.0
            r2 = 0.0
            if self.sig_downflow and i != 0 and i <= outlet_layer:
                r1 = self.vec2[i - 1] * self.timedelta / self.dz
            if not self.sig_downflow and i != self.num_layer - 1 and i >= outlet_layer:
                r2 = self.vec2[i + 1] * self.timedelta / self.dz

            self.cal_mat[0, i] = -(2*s + r1) if i == self.num_layer - 1 else (-(s + r1) if i != 0 else 0.0)
            self.cal_mat[2, i] = -(2*s + r2) if i == 0 else (-(s + r2) if i != self.num_layer - 1 else 0.0)
            self.cal_mat[1, i] = 2*s + r1 + r2 + 1 + p
            self.tes_temp[i] += p * self.t_ambient

            if self.vec1[i] != 0:
                q = self.timedelta * self.vec1[i] / self.base_area
                self.cal_mat[1, i] += q
                self.tes_temp[i] += q * self.tin

        self.tes_temp = self.TDMA_solver(
            self.cal_mat[0, 1:],
            self.cal_mat[1, :],
            self.cal_mat[2, :-1],
            self.tes_temp,
        )
        self.tout = self.tes_temp[outlet_layer]

        total = float(np.sum(self.tes_temp - self.t_ref))
        rho_ref = self.water_density(self.t_ref)
        self.heat = 0.001 * total * rho_ref * 4.186 * self.dz * self.base_area

        return self.tout


# ---------------------------------------------------------------------------
# HeatExchangerW2W
# ---------------------------------------------------------------------------

class HeatExchangerW2W:
    """Water-to-water plate heat exchanger with Dittus–Boelter correlation.

    Internal geometry is fixed (plate dimensions hard-coded).
    High-temperature and low-temperature sides are identified automatically.

    Attributes:
        flowrate_high: High-temperature-side flow rate [m³/min].
        flowrate_low: Low-temperature-side flow rate [m³/min].
        t_water_outlet_high: High-side outlet temperature [°C].
        t_water_outlet_low: Low-side outlet temperature [°C].
    """

    def __init__(self) -> None:
        """Initialize water-to-water heat exchanger state."""
        self.flowrate_high = 0.0
        self.flowrate_low = 0.0
        self.t_water_inlet_high = 0.0
        self.t_water_inlet_low = 0.0
        self.t_water_outlet_high = 0.0
        self.t_water_outlet_low = 0.0

    def cal(
        self,
        flowrate_high: float,
        flowrate_low: float,
        t_water_inlet_high: float,
        t_water_inlet_low: float,
        t_water_outlet_high: float,
        t_water_outlet_low: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate outlet temperatures using LMTD / Dittus–Boelter method.

        If the nominal "high" side is actually cooler, the sides are swapped
        internally for calculation and the result is swapped back.

        Args:
            flowrate_high: High-temperature-side flow rate [kg/s].
            flowrate_low: Low-temperature-side flow rate [kg/s].
            t_water_inlet_high: High-side inlet temperature [°C].
            t_water_inlet_low: Low-side inlet temperature [°C].
            t_water_outlet_high: High-side outlet temperature guess [°C].
            t_water_outlet_low: Low-side outlet temperature guess [°C].

        Returns:
            Tuple of ``(flowrate_high, flowrate_low, t_inlet_high, t_inlet_low,
            t_outlet_high, t_outlet_low)`` with flow rates in m³/min.
        """
        flowrate_high = flowrate_high / 1000 * 60
        flowrate_low = flowrate_low / 1000 * 60

        if flowrate_high < 0.01 or flowrate_low < 0.01:
            return (flowrate_high, flowrate_low,
                    t_water_inlet_high, t_water_inlet_low,
                    t_water_inlet_high, t_water_inlet_low)

        flag = 0
        if t_water_inlet_high < t_water_inlet_low:
            flag = 1
            t_water_inlet_high, t_water_inlet_low = t_water_inlet_low, t_water_inlet_high
            flowrate_high, flowrate_low = flowrate_low, flowrate_high

        cp = 1.0
        s = 0.01
        c = 40.0
        d = 1.0
        de = 2 * c * d / (c + d)
        area = 250.0

        def _htc(t_avg: float, flowrate: float) -> tuple[float, float]:
            gamma = (1.735e-5*t_avg**3 - 6.133e-3*t_avg**2 + 2.704e-2*t_avg + 1000.0)
            l = flowrate * 60 * gamma
            eta = (4.359e-8*t_avg**4 - 1.109e-5*t_avg**3 + 1.107e-3*t_avg**2
                   - 5.824e-2*t_avg + 1.826) * 1e-4
            mu = 3600.0 * 9.80 * eta
            lamb = -5.672e-6*t_avg**2 + 1.556e-3*t_avg + 0.4894
            htc = 0.023 * cp**(1/3) * l**0.8 * lamb**(2/3) * s**(-0.8) * mu**(-7/15) * de**(-0.2)
            w = flowrate / 60 * gamma * 4.186
            return htc, w

        t_avg_h = (t_water_inlet_high + t_water_outlet_high) / 2
        ah, wh = _htc(t_avg_h, flowrate_high)

        t_avg_l = (t_water_inlet_low + t_water_outlet_low) / 2
        ac, wc = _htc(t_avg_l, flowrate_low)

        k = ah * ac / (ah + ac) * 1.162e-3
        ka = k * area
        wcwh = wc / wh
        z = ka * (1 - wcwh) / wc

        if wcwh != 1:
            x = math.exp(z)
            y = t_water_inlet_high - t_water_inlet_low
            t_water_outlet_high = t_water_inlet_high - y * (1 - x) / (1 - x / wcwh)
            t_water_outlet_low = t_water_inlet_low + y * (1 - x) / (wcwh - x)

            if z > 200:
                if wc > wh:
                    t_water_outlet_high = t_water_inlet_low
                    t_water_outlet_low = (
                        (t_water_inlet_high - t_water_inlet_low) * wh / wc + t_water_inlet_low
                    )
                else:
                    t_water_outlet_low = t_water_inlet_high
                    t_water_outlet_high = (
                        (t_water_inlet_low - t_water_inlet_high) * wh / wc + t_water_inlet_high
                    )
        else:
            y = t_water_inlet_high - t_water_inlet_low
            t_water_outlet_high = t_water_inlet_high - y * ka / (wc + ka)
            t_water_outlet_low = t_water_inlet_low - y * ka / (wc + ka)

        if t_water_outlet_high < t_water_inlet_low or t_water_outlet_low > t_water_inlet_high:
            if flowrate_low > flowrate_high:
                t_water_outlet_high = t_water_inlet_low
                t_water_outlet_low = (
                    (t_water_outlet_high - t_water_inlet_high) * flowrate_high / flowrate_low
                    + t_water_inlet_low
                )
            else:
                t_water_outlet_low = t_water_inlet_high
                t_water_outlet_high = (
                    (t_water_outlet_low - t_water_inlet_low) * flowrate_low / flowrate_high
                    + t_water_inlet_high
                )

        if flag == 1:
            t_water_outlet_high, t_water_outlet_low = t_water_outlet_low, t_water_outlet_high
            flowrate_high, flowrate_low = flowrate_low, flowrate_high

        return (flowrate_high, flowrate_low,
                t_water_inlet_high, t_water_inlet_low,
                t_water_outlet_high, t_water_outlet_low)


# ---------------------------------------------------------------------------
# HeatExchangerW2A
# ---------------------------------------------------------------------------

class HeatExchangerW2A:
    """Water-to-air heat exchanger (cooling/heating coil).

    Handles dry, wet, and mixed (partial condensation) conditions.
    Surface area is calibrated from rated conditions at initialisation.

    References:
        HVACSIM+; 宇田川 (1986); 富樫 (2016).

    Attributes:
        area_surface: Total heat-transfer surface area [m²].
        rated_coef_dry: Rated dry overall HTC [kW/(m²·K)].
        rated_coef_wet: Rated wet overall HTC [kW/(m²·(kJ/kg))].
        q: Heat-exchange rate [kW].
        tout_air: Air outlet dry-bulb temperature [°C].
        wout_air: Air outlet absolute humidity [kg/kg'].
        tout_water: Water outlet temperature [°C].
        ratio_drywet: Dry-surface fraction (0 = fully wet, 1 = fully dry).
    """

    def __init__(
        self,
        rated_g_air: float = 2.5,
        rated_v_air: float = 2.99,
        rated_tdbin_air: float = 27.2,
        rated_twbin_air: float = 20.1,
        rated_g_water: float = 1.9833333,
        rated_v_water: float = 1.25,
        rated_tin_water: float = 7.0,
        rated_q: float = 40.4,
        rated_rh_border: float = 95.0,
    ) -> None:
        """Initialize coil and calibrate surface area from rated conditions.

        Args:
            rated_g_air: Rated air mass flow rate [kg/s].
            rated_v_air: Rated air face velocity [m/s].
            rated_tdbin_air: Rated air inlet dry-bulb temperature [°C].
            rated_twbin_air: Rated air inlet wet-bulb temperature [°C].
            rated_g_water: Rated water mass flow rate [kg/s].
            rated_v_water: Rated water velocity [m/s].
            rated_tin_water: Rated water inlet temperature [°C].
            rated_q: Rated heat-exchange capacity [kW].
            rated_rh_border: Boundary relative humidity for dry/wet split [%].
        """
        self.rated_v_water = rated_v_water
        self.rated_v_air = rated_v_air
        self.rated_g_air = rated_g_air
        self.rated_g_water = rated_g_water
        self.rated_tdbin_air = rated_tdbin_air
        self.rated_twbin_air = rated_twbin_air
        self.win_air = tdb_twb2w(self.rated_tdbin_air, self.rated_twbin_air)
        self.rated_tin_water = rated_tin_water
        self.rated_q = rated_q
        self.rated_rh_border = rated_rh_border

        self.tout_air = 20.0
        self.wout_air = 0.01
        self.rhout_air = 50.0
        self.rhin_air = 50.0
        self.tout_water = 10.0
        self.q = 0.0
        self.tdb_in_air = 20.0
        self.w_in_air = 0.001
        self.rh_border = self.rated_rh_border
        self.tin_water = 10.0
        self.g_air = 0.0
        self.g_water = 0.0
        self.ratio_drywet = 0.0
        self.coolingcoil = 0
        self.heatingcoil = 0

        self.rated_coef_dry = 1 / (
            4.72 + 4.91 * math.pow(self.rated_v_water, -0.8) + 26.7 * math.pow(self.rated_v_air, -0.64)
        )
        self.rated_coef_wet = 1 / (
            10.044 + 10.44 * math.pow(self.rated_v_water, -0.8) + 39.6 * math.pow(self.rated_v_air, -0.64)
        )

        self.rated_cpma = w2cpair(self.win_air)
        self.rated_cap_air = self.rated_g_air * self.rated_cpma
        self.rated_cap_water = self.rated_g_water * 4.186

        if self.rated_tdbin_air < self.rated_tin_water:
            # heating coil
            self.rated_cap_min = min(self.rated_cap_air, self.rated_cap_water)
            self.rated_cap_max = max(self.rated_cap_air, self.rated_cap_water)
            self.rated_eff = self.rated_q / self.rated_cap_min / (
                self.rated_tin_water - self.rated_tdbin_air
            )
            # Bug fix: original passed 4 args; hex_ntu takes ratio not separate min/max
            self.rated_ntu = hex_ntu(
                self.rated_eff, self.rated_cap_min / self.rated_cap_max, "counterflow"
            )
            self.area_surface = self.rated_ntu * self.rated_cap_min / self.rated_coef_dry
        else:
            # cooling coil
            self.rated_tout_water = self.rated_tin_water + self.rated_q / self.rated_cap_water
            self.rated_hin_air = tdb_w2h(self.rated_tdbin_air, self.win_air)
            self.rated_hout_air = self.rated_hin_air - self.rated_q / self.rated_g_air

            self.rated_wout_air = h_rh2w(self.rated_hout_air, self.rated_rh_border)

            if self.win_air < self.rated_wout_air:
                self.rated_tout_air = self.rated_tdbin_air - self.rated_q / self.rated_cap_air
                self.rated_d1 = self.rated_tdbin_air - self.rated_tout_water
                self.rated_d2 = self.rated_tout_air - self.rated_tin_water
                self.rated_lmtd = (self.rated_d1 - self.rated_d2) / math.log(
                    self.rated_d1 / self.rated_d2
                )
                self.area_surface = self.rated_q / self.rated_lmtd / self.rated_coef_dry
            else:
                self.rated_t_border_air = w_rh2tdb(self.win_air, self.rated_rh_border)
                self.rated_h_border_air = tdb_w2h(self.rated_t_border_air, self.win_air)

                self.rated_h_border_wet = (
                    (self.rated_h_border_air - self.rated_hout_air) * self.rated_g_air
                )
                self.rated_t_border_water = (
                    self.rated_tin_water + self.rated_h_border_wet / self.rated_cap_water
                )
                self.rated_h_water_inlet = tdb2hsat(self.rated_tin_water)
                self.rated_h_border_water = tdb2hsat(self.rated_t_border_water)

                self.rated_dt1 = self.rated_tdbin_air - self.rated_tout_water
                self.rated_dt2 = self.rated_t_border_air - self.rated_t_border_water
                self.rated_lmtd = (self.rated_dt1 - self.rated_dt2) / math.log(
                    self.rated_dt1 / self.rated_dt2
                )
                self.rated_area_dry_sur = (
                    (self.rated_q - self.rated_h_border_wet) / self.rated_lmtd / self.rated_coef_dry
                )
                self.rated_dh1 = self.rated_h_border_air - self.rated_h_border_water
                self.rated_dh2 = self.rated_hout_air - self.rated_h_water_inlet
                self.rated_lmhd = (self.rated_dh1 - self.rated_dh2) / math.log(
                    self.rated_dh1 / self.rated_dh2
                )
                self.rated_area_wet_sur = (
                    self.rated_h_border_wet / self.rated_lmhd / self.rated_coef_wet
                )
                self.area_surface = self.rated_area_dry_sur + self.rated_area_wet_sur

    def cal(
        self,
        tdb_in_air: float,
        w_in_air: float,
        tin_water: float,
        g_air: float,
        g_water: float,
    ) -> tuple:
        """Calculate coil outlet conditions.

        Args:
            tdb_in_air: Air inlet dry-bulb temperature [°C].
            w_in_air: Air inlet absolute humidity [kg/kg'].
            tin_water: Water inlet temperature [°C].
            g_air: Air mass flow rate [kg/s].
            g_water: Water mass flow rate [kg/s].

        Returns:
            Tuple of ``(tout_air, wout_air, rhout_air, tout_water,
            ratio_drywet, q, g_water, area_surface)``.
        """
        self.tdb_in_air = tdb_in_air
        self.w_in_air = w_in_air
        self.rhin_air = w_tdb2rh(self.w_in_air, self.tdb_in_air)
        self.tin_water = tin_water
        self.g_air = g_air
        self.g_water = g_water
        self.ratio_drywet = 0.0
        self.coolingcoil = 0
        self.heatingcoil = 0

        self.t_border_water = self.t_border_air = 0.0
        self.zd = self.wd = self.v1 = self.v2 = 0.0
        self.zw = self.ww = self.v3 = self.v4 = self.v5 = self.v6 = 0.0

        if self.g_air <= 0 or self.g_water <= 0 or self.tdb_in_air == self.tin_water:
            self.tout_air = self.tdb_in_air
            self.tout_water = self.tin_water
            self.wout_air = self.w_in_air
            self.rhout_air = w_tdb2rh(self.wout_air, self.tout_air)
            self.ratio_drywet = 1.0
            self.q = 0.0
            return (self.tout_air, self.wout_air, self.rhout_air,
                    self.tout_water, self.ratio_drywet, self.q, self.g_water)

        self.cpma = w2cpair(self.w_in_air)
        self.cap_air = self.g_air * self.cpma
        self.cap_water = self.g_water * 4.186

        self.v_water = (self.g_water / self.rated_g_water) * self.rated_v_water
        self.v_air = (self.g_air / self.rated_g_air) * self.rated_v_air
        self.coef_dry = 1 / (
            4.72 + 4.91 * math.pow(self.v_water, -0.8) + 26.7 * math.pow(self.v_air, -0.64)
        )
        self.coef_wet = 1 / (
            10.044 + 10.44 * math.pow(self.v_water, -0.8) + 39.6 * math.pow(self.v_air, -0.64)
        )

        if self.tdb_in_air < self.tin_water:
            # Heating coil
            self.heatingcoil = 1
            self.ratio_drywet = 1.0
            cap_min = min(self.cap_air, self.cap_water)
            cap_max = max(self.cap_air, self.cap_water)
            ntu = self.coef_dry * self.area_surface / cap_min
            eff = hex_effectiveness(ntu, cap_min / cap_max, "counterflow")
            self.q = eff * cap_min * (self.tin_water - self.tdb_in_air)
            self.tout_air = self.tdb_in_air + self.q / self.cap_air
            self.tout_water = self.tin_water - self.q / self.cap_water
            self.wout_air = self.w_in_air
            self.rhout_air = w_tdb2rh(self.wout_air, self.tout_air)
        else:
            # Cooling coil
            self.coolingcoil = 1
            t_border = w_rh2tdb(self.w_in_air, self.rh_border)
            hin_air = tdb_w2h(self.tdb_in_air, self.w_in_air)
            [p_a, p_b] = getparameter_hex(self.tin_water)

            xd = 1 / self.cap_air
            yd = -1 / self.cap_water
            xw = 1 / self.g_air
            yw = -p_a / self.cap_water

            def efunc(fdrate: float) -> float:
                self.zd = math.exp(self.coef_dry * self.area_surface * fdrate * (xd + yd))
                self.wd = self.zd * xd + yd
                self.v1 = xd * (self.zd - 1) / self.wd
                self.v2 = self.zd * (xd + yd) / self.wd
                self.zw = math.exp(self.coef_wet * self.area_surface * (1 - fdrate) * (xw + yw))
                self.ww = self.zw * xw + yw
                self.v3 = (xw + yw) / self.ww
                self.v4 = xw * (self.zw - 1) / self.ww
                self.v5 = self.zw * (xw + yw) / self.ww
                self.v6 = yw * (1 - self.zw) / self.ww / p_a
                self.t_border_water = (
                    self.v5 * self.tin_water
                    + self.v6 * (hin_air - self.v1 * self.cpma * self.tdb_in_air - p_b)
                ) / (1 - self.v1 * self.v6 * self.cpma)
                self.t_border_air = self.tdb_in_air - self.v1 * (
                    self.tdb_in_air - self.t_border_water
                )
                if fdrate == 1 and self.t_border_air > t_border:
                    return 1.0
                return t_border - self.t_border_air

            drate = 1.0
            if 0 < efunc(drate):
                try:
                    drate = optimize.brentq(efunc, 0, 1, xtol=0.0001, maxiter=100)
                except ValueError:
                    drate = 0.0
                    efunc(drate)

            self.tout_water = self.tdb_in_air - self.v2 * (
                self.tdb_in_air - self.t_border_water
            )
            h_border_air = self.cpma * (self.t_border_air - self.tdb_in_air) + hin_air
            h_water_inlet = p_a * self.tin_water + p_b
            hout_air = self.v3 * h_border_air + self.v4 * h_water_inlet
            self.q = (
                self.cap_air * (self.tdb_in_air - self.t_border_air)
                + self.g_air * (h_border_air - hout_air)
            )
            self.ratio_drywet = drate

            if drate < 1:
                self.wout_air = h_rh2w(hout_air, self.rh_border)
            else:
                self.wout_air = self.w_in_air

            self.tout_air = w_h2tdb(self.wout_air, hout_air)
            self.rhout_air = w_tdb2rh(self.wout_air, self.tout_air)

        return (self.tout_air, self.wout_air, self.rhout_air,
                self.tout_water, self.ratio_drywet, self.q,
                self.g_water, self.area_surface)


# ---------------------------------------------------------------------------
# Damper
# ---------------------------------------------------------------------------

class Damper:
    """Air damper with linearly interpolated pressure-loss characteristic.

    Attributes:
        coef: List of ``[opening, coefficient]`` pairs defining the characteristic.
        damp: Current damper opening (0.0 = fully closed, 1.0 = fully open).
        g: Air flow rate [m³/min].
        dp: Pressure drop [Pa].
    """

    def __init__(
        self,
        coef: list[list[float]] | None = None,
    ) -> None:
        """Initialize damper with opening–coefficient table.

        Args:
            coef: List of ``[opening, a]`` pairs (descending opening order).
                  Pressure drop is computed as ``dp = a * g^2``.
                  Default: standard 6-point characteristic.
        """
        if coef is None:
            coef = [
                [1.0, 0.020], [0.8, 0.2],
                [0.6, 1.0], [0.4, 2.0], [0.2, 5.0], [0.0, 999.9],
            ]
        self.coef = coef
        self.g = 0.0
        self.damp = 0.0
        self.dp = 0.0

    def f2p(self, g: float) -> float:
        """Calculate pressure drop from flow rate.

        Args:
            g: Air flow rate [m³/min].

        Returns:
            Pressure drop [Pa] (negative convention).
        """
        n = len(self.coef)
        self.g = g
        if self.damp >= self.coef[0][0]:
            coef_val = self.coef[0][1] * self.g ** 2
        elif self.damp <= self.coef[n - 1][0]:
            coef_val = self.coef[n - 1][1] * self.g ** 2
        else:
            for i in range(1, n):
                coef_a = self.coef[i - 1]
                coef_b = self.coef[i]
                if coef_b[0] <= self.damp < coef_a[0]:
                    break
            a = coef_a[1] * self.g ** 2
            b = coef_b[1] * self.g ** 2
            coef_val = (a - b) / (coef_a[0] - coef_b[0]) * (self.damp - coef_b[0]) + b

        self.dp = -coef_val
        return self.dp

    def p2f(self, dp: float) -> float:
        """Calculate flow rate from pressure drop.

        Args:
            dp: Pressure drop [Pa].

        Returns:
            Flow rate [m³/min].
        """
        n = len(self.coef)
        self.dp = abs(dp)
        if self.damp >= self.coef[0][0]:
            self.g = pow(self.dp / self.coef[0][1], 0.5)
        elif self.damp <= self.coef[n - 1][0]:
            self.g = pow(self.dp / self.coef[n - 1][1], 0.5)
        else:
            for i in range(1, n):
                coef_a = self.coef[i - 1]
                coef_b = self.coef[i]
                if coef_b[0] <= self.damp < coef_a[0]:
                    break
            a = pow(self.dp / coef_a[1], 0.5)
            b = pow(self.dp / coef_b[1], 0.5)
            self.g = (a - b) / (coef_a[0] - coef_b[0]) * (self.damp - coef_a[0]) + a

        self.g = self.g if dp >= 0 else -self.g
        return self.g


# ---------------------------------------------------------------------------
# Fan
# ---------------------------------------------------------------------------

class Fan:
    """Centrifugal fan with cubic pressure-flow and efficiency curves.

    Supports INV speed control. Fan unit pressure is in Pa (not kPa).

    Attributes:
        pg: Pressure-flow curve coefficients [c0, c1, c2, c3].
        eg: Efficiency-flow curve coefficients [c0, c1, c2].
        r_ef: Rated maximum efficiency.
        inv: INV frequency ratio (0.0–1.0).
        g_d: Rated air flow rate [m³/min].
        dp: Fan static pressure [Pa].
        g: Air flow rate [m³/min].
        ef: Fan efficiency.
        pw: Power consumption [kW].
        flag: 0 = normal, 1 = pressure clipped, 2 = zero efficiency.
    """

    def __init__(
        self,
        pg: list[float] | None = None,
        eg: list[float] | None = None,
        r_ef: float = 0.6,
        g_d: float = 80.0,
        inv: float = 1.0,
        figure: int = 1,
    ) -> None:
        """Initialize fan with performance curve parameters.

        Args:
            pg: Pressure-flow coefficients [c0, c1, c2, c3].
                Default: ``[948.66, -0.5041, -0.097, 0]``.
            eg: Efficiency-flow coefficients [c0, c1, c2].
                Default: ``[0.0773, 0.0142, -1.00e-04]``.
            r_ef: Rated maximum efficiency. Default: 0.6.
            g_d: Rated air flow rate [m³/min]. Default: 80.
            inv: INV frequency ratio (0.0–1.0). Default: 1.0.
            figure: If 1, display performance curves on instantiation.
        """
        (_, _, _, text) = traceback.extract_stack()[-2]
        self.name = text[: text.find("=")].strip()

        self.pg = [948.66, -0.5041, -0.097, 0] if pg is None else pg
        self.eg = [0.0773, 0.0142, -1.00e-04] if eg is None else eg
        self.r_ef = r_ef
        self.inv = inv
        self.dp = 0.0
        self.g = 0.0
        self.ef = 0.0
        self.pw = 0.0
        self.flag = 0
        self.num = 1
        self.g_d = g_d

        if figure == 1:
            self.figure_curve()

    def f2p(self, g: float) -> float:
        """Calculate fan static pressure from flow rate.

        Args:
            g: Air flow rate [m³/min].

        Returns:
            Fan static pressure [Pa].
        """
        self.g = g
        if self.g > 0:
            self.dp = (
                self.pg[0]
                + self.pg[1] * (self.g / self.inv)
                + self.pg[2] * (self.g / self.inv) ** 2
                + self.pg[3] * (self.g / self.inv) ** 3
            ) * self.inv ** 2
        else:
            self.dp = 0.0

        if self.dp < 0:
            self.dp = 0.0
            self.flag = 1
        else:
            self.flag = 0
        return self.dp

    def f2p_co(self) -> list[float]:
        """Return polynomial coefficients adjusted for current INV ratio.

        Returns:
            Coefficients ``[c0, c1, c2, c3]`` for dp = c0 + c1*g + c2*g² + c3*g³.
        """
        if self.inv > 0:
            return [
                self.pg[0] * self.inv ** 2,
                self.pg[1] * self.inv,
                self.pg[2],
                self.pg[3] / self.inv,
            ]
        return [0.0, 0.0, 0.0, 0.0]

    def cal(self) -> None:
        """Calculate power consumption and efficiency at the current operating point.

        Updates *pw*, *ef*, *dp*, and *flag* in place.
        """
        if self.g > 0 and self.inv > 0:
            G = self.g / self.inv
            K = (1.0 - (1.0 - self.r_ef) / (self.inv ** 0.2)) / self.r_ef
            self.ef = K * (self.eg[0] + self.eg[1] * G + self.eg[2] * G ** 2)
            self.dp = (
                self.pg[0]
                + self.pg[1] * (self.g / self.inv)
                + self.pg[2] * (self.g / self.inv) ** 2
                + self.pg[3] * (self.g / self.inv) ** 3
            ) * self.inv ** 2
            if self.dp < 0:
                self.dp = 0.0
                self.flag = 1
            if self.ef > 0:
                self.pw = self.g * (self.dp / 1000) / (60 * 0.8 * self.ef)
                self.flag = 0
            else:
                self.pw = 0.0
                self.flag = 2
        else:
            self.pw = 0.0
            self.flag = 0

    def figure_curve(self) -> None:
        """Display pressure-flow and efficiency-flow curves using matplotlib."""
        quadratic_formula(self.pg[0], self.pg[1], self.pg[2])
        x = np.linspace(0, self.g_d, 50)
        y_p = (
            self.pg[0]
            + self.pg[1] * (x / self.inv)
            + self.pg[2] * (x / self.inv) ** 2
        ) * self.inv ** 2
        x2 = np.linspace(0, self.g_d * self.inv, 50)
        k = (1.0 - (1.0 - self.r_ef) / (self.inv ** 0.2)) / self.r_ef
        y_e = k * (self.eg[0] + self.eg[1] * (x2 / self.inv) + self.eg[2] * (x2 / self.inv) ** 2)
        fig, ax1 = plt.subplots()
        color1 = "tab:orange"
        ax1.set_xlabel("Flow [m3/min]")
        ax1.set_ylabel("Pressure [Pa]", color=color1)
        ax1.plot(x, y_p, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, self.pg[0] + 10)
        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("Efficiency [-]", color=color2)
        ax2.plot(x2, y_e, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 1)
        plt.title("{}".format(self.name))
        plt.show()


# ---------------------------------------------------------------------------
# SteamSprayHumidifier
# ---------------------------------------------------------------------------

class SteamSprayHumidifier:
    """Steam-spray humidifier modeled as a constant-pressure mixing process.

    The outlet humidity is limited by saturation efficiency and the saturated
    absolute humidity at the outlet air temperature.

    References:
        HVACSIM+ TYPE 22 (NIST IR 84-2996, p.90).
        TRNSYS aux_heat_and_cool/616new.for.

    Attributes:
        area_humidifier: Humidifier cross-sectional area [m²].
        dp: Pressure drop [Pa].
        eff: Saturation efficiency (0–1).
        tdb_air_out: Outlet air dry-bulb temperature [°C].
        w_air_out: Outlet air absolute humidity [kg/kg'].
        flowrate_air_out: Outlet air mass flow rate [kg/s].
    """

    def __init__(
        self,
        area_humidifier: float = 0.42,
        dp: float = 45.0,
        eff: float = 1.0,
    ) -> None:
        """Initialize steam-spray humidifier.

        Args:
            area_humidifier: Humidifier face area [m²].
            dp: Pressure drop [Pa].
            eff: Saturation efficiency (0–1). Default: 1.0.
        """
        self.area_humidifier = area_humidifier
        self.dp = dp
        self.eff = eff

        self.tdb_air_in = 0.0
        self.w_air_in = 0.0
        self.flowrate_air_in = 0.0
        self.flowrate_steam_in = 0.0
        self.t_steam_in = 0.0
        self.tdb_air_out = 0.0
        self.w_air_out = 0.0
        self.flowrate_air_out = 0.0

    def cal(
        self,
        tdb_air_in: float,
        w_air_in: float,
        flowrate_air_in: float,
        flowrate_steam_in: float,
        t_steam_in: float,
    ) -> tuple[float, float, float]:
        """Calculate humidifier outlet conditions.

        Args:
            tdb_air_in: Inlet air dry-bulb temperature [°C].
            w_air_in: Inlet air absolute humidity [kg/kg'].
            flowrate_air_in: Inlet air mass flow rate [kg/s].
            flowrate_steam_in: Steam injection mass flow rate [kg/s].
            t_steam_in: Steam inlet temperature [°C].

        Returns:
            Tuple of ``(tdb_air_out, w_air_out, flowrate_air_out)``.
        """
        self.tdb_air_in = tdb_air_in
        self.w_air_in = w_air_in
        self.flowrate_air_in = flowrate_air_in
        self.flowrate_steam_in = flowrate_steam_in
        self.t_steam_in = t_steam_in

        Cpa = 1.006   # dry-air specific heat [kJ/(kg·K)]
        Cps = 1.86    # steam specific heat [kJ/(kg·K)]
        Cpai = (Cpa + self.w_air_in * Cps) / (1 + self.w_air_in)

        self.tdb_air_out = (
            self.flowrate_steam_in * Cps * self.t_steam_in
            + self.flowrate_air_in * Cpai * self.tdb_air_in
        ) / (self.flowrate_steam_in * Cps + self.flowrate_air_in * Cpai)

        w_air_out = self.w_air_in + self.flowrate_steam_in * (1 + self.w_air_in) / self.flowrate_air_in

        # Bug fix: was psy_psat_tsat / psy_w_pv (undefined in original)
        p_sat = tdp2psat(self.tdb_air_out)
        w_sat = pv2w(p_sat)

        self.w_air_out = min(w_air_out, self.eff * w_sat)
        self.flowrate_air_out = (
            self.flowrate_air_in * (1 + self.w_air_out) / (1 + self.w_air_in)
        )

        return self.tdb_air_out, self.w_air_out, self.flowrate_air_out
