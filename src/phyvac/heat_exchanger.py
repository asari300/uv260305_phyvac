# -*- coding: utf-8 -*-
"""熱交換器の基本計算モジュール (Heat Exchanger).

NTU-効率法に基づく熱交換器計算。

Reference:
    宇田川光弘：パソコンによる空気調和計算法，オーム社，p.8-219，1986年.
"""

import math

from scipy import optimize

from .psychrometrics import tdb2hsat


def getparameter_hex(tdb: float) -> tuple[float, float]:
    """熱交換器計算に必要な飽和空気比エンタルピーの線形近似パラメータを取得する。

    飽和空気比エンタルピー hws を tdb の一次関数 hws = fa * tdb + fb で近似する。

    Args:
        tdb: 乾球温度 [°C]

    Returns:
        (fa, fb) のタプル。
            fa: 飽和空気比エンタルピーの温度微分 [kJ/(kg'·K)]
            fb: 切片 [kJ/kg']
    """
    delta = 0.001
    hws1 = tdb2hsat(tdb)
    hws2 = tdb2hsat(tdb + delta)
    fa = (hws2 - hws1) / delta
    fb = hws1 - fa * tdb
    return fa, fb


def hex_effectiveness(ntu: float, ratio_heat_cap: float, flowtype: str) -> float:
    """熱交換器の熱通過有効度を計算する。

    Args:
        ntu:            熱通過数 (Number of Transfer Units) [-]
        ratio_heat_cap: 熱容量比 Cmin/Cmax [-] (0 ~ 1)
        flowtype:       流れ形式。'counterflow' または 'parallelflow'

    Returns:
        熱通過有効度 [-] (0 ~ 1)

    Raises:
        ValueError: flowtype が不正な値の場合
    """
    ratio = ratio_heat_cap

    if ratio <= 0:
        return 1 - math.exp(-ntu)
    if ntu == 0:
        return 0.0

    if flowtype == "counterflow":
        if ratio < 1:
            return (1 - math.exp((ratio - 1) * ntu)) / (
                1 - ratio * math.exp((ratio - 1) * ntu)
            )
        else:
            return ntu / (1 + ntu)
    elif flowtype == "parallelflow":
        if ratio < 1:
            return (1 - math.exp(-ntu * (ratio + 1))) / (1 + ratio)
        else:
            return 0.5 * (1 - math.exp(-2 * ntu))
    else:
        raise ValueError(f"flowtype は 'counterflow' または 'parallelflow' を指定してください。得: {flowtype!r}")


def hex_ntu(feff: float, fratio_heat_cap: float, fflowtype: str) -> float:
    """熱交換器の NTU を計算する (有効度から逆算)。

    Args:
        feff:            目標とする熱通過有効度 [-]
        fratio_heat_cap: 熱容量比 Cmin/Cmax [-]
        fflowtype:       流れ形式。'counterflow' または 'parallelflow'

    Returns:
        熱通過数 NTU [-]
    """

    def _residual(fntu: float) -> float:
        return feff - hex_effectiveness(fntu, fratio_heat_cap, fflowtype)

    return optimize.newton(_residual, x0=0.0, tol=1e-6, maxiter=20)
