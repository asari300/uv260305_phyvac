# -*- coding: utf-8 -*-
"""空気の熱・湿り空気状態計算モジュール (Psychrometrics).

空気調和・衛生工学会編：空気調和・衛生工学便覧14版，1基礎編，第3章, pp.39-56，2010.

記号の定義:
    twb:    湿球温度 [°C]
    tdb:    乾球温度 [°C]
    tdp:    露点温度 [°C]
    w:      絶対湿度 [kg/kg']
    pv:     水蒸気分圧 [kPa]
    psat:   飽和空気の水蒸気分圧 [kPa]
    h:      比エンタルピー [kJ/kg']
    rh:     相対湿度 [%]
    den:    密度 [kg/m^3]
    p_atm:  標準大気圧 [kPa] (= 101.325 kPa)
"""

import math

from scipy import optimize

# 乾き空気の定圧比熱 [kJ/(kg·K)]
CA: float = 1.006
# 水蒸気の定圧比熱 [kJ/(kg·K)]
CV: float = 1.86
# 0°C の水の蒸発潜熱 [kJ/kg]
R0: float = 2.501e3


def tdb_rh2tdp(tdb: float, rh: float) -> float:
    """乾球温度と相対湿度から露点温度を計算する。

    Args:
        tdb: 乾球温度 [°C]
        rh:  相対湿度 [%] (0 ~ 100)

    Returns:
        露点温度 [°C]
    """
    c = 373.16 / (273.16 + tdb)
    b = c - 1
    a = (
        -7.90298 * b
        + 5.02808 * math.log10(c)
        - 1.3816e-7 * (10 ** (11.344 * b / c) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * b) - 1)
    )
    psat = 760 * 10 ** a

    x = 0.622 * (rh * psat) / 100 / (760 - rh * psat / 100)
    psat = 100 * 760 * x / (100 * (0.622 + x))

    psat0 = 0.0
    tdp_max = tdb
    tdp_min = -20.0
    tdp = 0.0
    cnt = 0

    while (psat - psat0 < -0.01) or (psat - psat0 > 0.01):
        tdp = (tdp_max + tdp_min) / 2
        c = 373.16 / (273.16 + tdp)
        b = c - 1
        a = (
            -7.90298 * b
            + 5.02808 * math.log10(c)
            - 1.3816e-7 * (10 ** (11.344 * b / c) - 1)
            + 8.1328e-3 * (10 ** (-3.49149 * b) - 1)
        )
        psat0 = 760 * 10 ** a

        if psat - psat0 > 0:
            tdp_min = tdp
        else:
            tdp_max = tdp

        cnt += 1
        if cnt > 30:
            break

    return tdp


def tdb_rh2h_x(tdb: float, rh: float) -> list[float]:
    """乾球温度と相対湿度から比エンタルピーと絶対湿度を計算する。

    Args:
        tdb: 乾球温度 [°C]
        rh:  相対湿度 [%] (0 ~ 100)

    Returns:
        [h, w] のリスト。
            h: 比エンタルピー [kJ/kg']
            w: 絶対湿度 [kg/kg']
    """
    c = 373.16 / (273.16 + tdb)
    b = c - 1
    a = (
        -7.90298 * b
        + 5.02808 * math.log10(c)
        - 1.3816e-7 * (10 ** (11.344 * b / c) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * b) - 1)
    )
    psat = 760 * 10 ** a
    w = 0.622 * (rh * psat) / 100 / (760 - rh * psat / 100)
    h = CA * tdb + (R0 + CV * tdb) * w
    return [h, w]


def tdb2psat(tdb: float) -> float:
    """乾球温度から飽和水蒸気圧を計算する (Wagner equation)。

    Args:
        tdb: 乾球温度 [°C]

    Returns:
        飽和水蒸気圧 [kPa]
    """
    x = 1 - (tdb + 273.15) / 647.3
    psat = 221200 * math.exp(
        (-7.76451 * x + 1.45838 * x**1.5 - 2.7758 * x**3 - 1.23303 * x**6) / (1 - x)
    )  # [hPa]
    return psat / 10  # [hPa] -> [kPa]


def tdb_rh2twb(tdb: float, rh: float) -> float:
    """乾球温度と相対湿度から湿球温度を計算する (Sprung equation)。

    Args:
        tdb: 乾球温度 [°C]
        rh:  相対湿度 [%] (0 ~ 100)

    Returns:
        湿球温度 [°C]
    """
    psat = tdb2psat(tdb)
    pv_1 = rh / 100 * psat
    pv_2 = -99999.0
    twb_max = 50.0
    twb_min = -50.0
    twb = 0.0
    cnt = 0

    while abs(pv_1 - pv_2) > 0.01:
        twb = (twb_max + twb_min) / 2
        pv_2 = tdb2psat(twb) - 0.000662 * 101.325 * (tdb - twb)

        if pv_1 - pv_2 > 0:
            twb_min = twb
        else:
            twb_max = twb

        cnt += 1
        if cnt > 20:
            break

    return twb


def tdb_w2h(tdb: float, w: float) -> float:
    """乾球温度と絶対湿度から比エンタルピーを計算する。

    Args:
        tdb: 乾球温度 [°C]
        w:   絶対湿度 [kg/kg']

    Returns:
        比エンタルピー [kJ/kg']
    """
    return CA * tdb + (CV * tdb + R0) * w


def tdb2hsat(tdb: float) -> float:
    """乾球温度から飽和湿り空気の比エンタルピーを計算する。

    Args:
        tdb: 乾球温度 [°C]

    Returns:
        飽和湿り空気の比エンタルピー [kJ/kg']
    """
    psat = tdp2psat(tdb)
    wsat = pv2w(psat)
    return tdb_w2h(tdb, wsat)


def w2pv(w: float, p_atm: float = 101.325) -> float:
    """絶対湿度から水蒸気分圧を計算する。

    Args:
        w:     絶対湿度 [kg/kg']
        p_atm: 大気圧 [kPa] (デフォルト: 101.325)

    Returns:
        水蒸気分圧 [kPa]
    """
    return p_atm * w / (0.622 + w)


def pv2w(pv: float, p_atm: float = 101.325) -> float:
    """水蒸気分圧から絶対湿度を計算する。

    Args:
        pv:    水蒸気分圧 [kPa]
        p_atm: 大気圧 [kPa] (デフォルト: 101.325)

    Returns:
        絶対湿度 [kg/kg']
    """
    return 0.622 * pv / (p_atm - pv)


def tdp2psat(tdp: float) -> float:
    """露点温度から飽和水蒸気圧を計算する。

    ASHRAE Fundamentals (2009) の式に基づく。

    Args:
        tdp: 露点温度 [°C]

    Returns:
        飽和水蒸気圧 [kPa]
    """
    p_convert = 0.001
    t = tdp + 273.15

    if tdp < 0.01:
        c1 = -5.6745359e3
        c2 = 6.3925247
        c3 = -9.6778430e-3
        c4 = 6.2215701e-7
        c5 = 2.0747825e-9
        c6 = -9.4840240e-13
        c7 = 4.1635019
        psat = math.exp(
            c1 / t + c2 + c3 * t + c4 * t**2 + c5 * t**3 + c6 * t**4 + c7 * math.log(t)
        ) * p_convert
    else:
        n1 = 0.11670521452767e4
        n2 = -0.72421316703206e6
        n3 = -0.17073846940092e2
        n4 = 0.12020824702470e5
        n5 = -0.32325550322333e7
        n6 = 0.14915108613530e2
        n7 = -0.4823265731591e4
        n8 = 0.40511340542057e6
        n9 = -0.23855557567849e0
        n10 = 0.65017534844798e3
        alpha = t + n9 / (t - n10)
        a2 = alpha**2
        a = a2 + n1 * alpha + n2
        b = n3 * a2 + n4 * alpha + n5
        c = n6 * a2 + n7 * alpha + n8
        psat = pow(2 * c / (-b + pow(b**2 - 4 * a * c, 0.5)), 4) / p_convert

    return psat


def h_rh2w(h: float, rh: float) -> float:
    """比エンタルピーと相対湿度から絶対湿度を計算する。

    Args:
        h:  比エンタルピー [kJ/kg']
        rh: 相対湿度 [%] (0 ~ 100)

    Returns:
        絶対湿度 [kg/kg']
    """
    tdb = h_rh2tdb(h, rh)
    return tdb_rh2w(tdb, rh)


def tdb2den(tdb: float) -> float:
    """乾球温度から乾き空気密度を計算する。

    Args:
        tdb: 乾球温度 [°C]

    Returns:
        密度 [kg/m^3]
    """
    return 1.293 * 273.3 / (273.2 + tdb)


def h_rh2tdb(h: float, rh: float) -> float:
    """比エンタルピーと相対湿度から乾球温度を計算する。

    Args:
        h:  比エンタルピー [kJ/kg']
        rh: 相対湿度 [%] (0 ~ 100)

    Returns:
        乾球温度 [°C]
    """

    def _residual(tdb: float) -> float:
        return h - tdb_rh2h(tdb, rh)

    return optimize.newton(_residual, x0=1e-5, tol=1e-4, maxiter=20)


def tdb_rh2h(tdb: float, rh: float) -> float:
    """乾球温度と相対湿度から比エンタルピーを計算する。

    Args:
        tdb: 乾球温度 [°C]
        rh:  相対湿度 [%] (0 ~ 100)

    Returns:
        比エンタルピー [kJ/kg']
    """
    w = tdb_rh2w(tdb, rh)
    return tdb_w2h(tdb, w)


def tdb_rh2w(tdb: float, rh: float) -> float:
    """乾球温度と相対湿度から絶対湿度を計算する。

    Args:
        tdb: 乾球温度 [°C]
        rh:  相対湿度 [%] (0 ~ 100)

    Returns:
        絶対湿度 [kg/kg']
    """
    psat = tdp2psat(tdb)
    pv = 0.01 * rh * psat
    return pv2w(pv)


def psat2tdp(psat: float) -> float:
    """飽和水蒸気圧から露点温度を計算する。

    ASHRAE Fundamentals (2009) の式に基づく。

    Args:
        psat: 飽和水蒸気圧 [kPa]

    Returns:
        露点温度 [°C]
    """
    p_convert = 0.001

    if psat < 0.611213:
        d1 = -6.0662e1
        d2 = 7.4624
        d3 = 2.0594e-1
        d4 = 1.6321e-2
        y = math.log(psat / p_convert)
        tdp = d1 + y * (d2 + y * (d3 + y * d4))
    else:
        n1 = 0.11670521452767e4
        n2 = -0.72421316703206e6
        n3 = -0.17073846940092e2
        n4 = 0.12020824702470e5
        n5 = -0.32325550322333e7
        n6 = 0.14915108613530e2
        n7 = -0.4823265731591e4
        n8 = 0.40511340542057e6
        n9 = -0.23855557567849e0
        n10 = 0.65017534844798e3
        ps = psat * p_convert
        beta = pow(ps, 0.25)
        b2 = beta**2
        e = b2 + n3 * beta + n6
        f = n1 * b2 + n4 * beta + n7
        g = n2 * b2 + n5 * beta + n8
        d = 2 * g / (-f - pow(f**2 - 4 * e * g, 0.5))
        tdp = (n10 + d - pow((n10 + d) ** 2 - 4 * (n9 + n10 * d), 0.5)) / 2 - 273.15

    return tdp


def w_h2tdb(w: float, h: float) -> float:
    """絶対湿度と比エンタルピーから乾球温度を計算する。

    Args:
        w: 絶対湿度 [kg/kg']
        h: 比エンタルピー [kJ/kg']

    Returns:
        乾球温度 [°C]
    """
    return (h - 2501 * w) / (1.006 + 1.86 * w)


def w_rh2tdb(w: float, rh: float) -> float:
    """絶対湿度と相対湿度から乾球温度を計算する。

    Args:
        w:  絶対湿度 [kg/kg']
        rh: 相対湿度 [%] (0 ~ 100)

    Returns:
        乾球温度 [°C]
    """
    psat = w2pv(w)
    return psat2tdp(psat / rh * 100)


def w2cpair(w: float) -> float:
    """絶対湿度から湿り空気の定圧比熱を計算する。

    Args:
        w: 絶対湿度 [kg/kg']

    Returns:
        定圧比熱 [kJ/(kg·K)]
    """
    return 1.006 + 1.86 * w


def w_tdb2rh(w: float, tdb: float) -> float:
    """絶対湿度と乾球温度から相対湿度を計算する。

    Args:
        w:   絶対湿度 [kg/kg']
        tdb: 乾球温度 [°C]

    Returns:
        相対湿度 [%] (0 ~ 100)。飽和水蒸気圧が 0 以下の場合は 0 を返す。
    """
    pv = w2pv(w)
    psat = tdp2psat(tdb)
    if psat <= 0:
        return 0.0
    return pv / psat * 100


def tdb_twb2w(tdb: float, twb: float) -> float:
    """乾球温度と湿球温度から絶対湿度を計算する。

    Args:
        tdb: 乾球温度 [°C]
        twb: 湿球温度 [°C]

    Returns:
        絶対湿度 [kg/kg']
    """
    psat = tdp2psat(twb)
    wsat = pv2w(psat)
    # 水の比熱 4.186 kJ/(kg·K)
    a = wsat * (2501 + (1.86 - 4.186) * twb) + 1.006 * (twb - tdb)
    b = 2501 + 1.86 * tdb - 4.186 * twb
    return a / b
