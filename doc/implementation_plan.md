# phyvac パッケージ リファクタリング実装方針

## 概要

`phyvac.py`（3770行のモノリシックファイル）を、責務ごとに分割したPythonパッケージへリファクタリングする。

---

## 現状

| 項目 | 内容 |
|------|------|
| ソースファイル | `phyvac.py`（単一ファイル、3770行） |
| パッケージ管理 | `uv` |
| テスト | なし |
| ドキュメント | インラインコメントのみ |

---

## 目標

| 項目 | 内容 |
|------|------|
| パッケージ構造 | `src/` レイアウト（PEP 517準拠） |
| モジュール分割 | 責務別に5モジュールへ分割 |
| ドキュメント | Google Style docstring（全関数・クラス・メソッド） |
| テスト | pytest によるユニットテスト |
| 依存管理 | `uv add` によるサードパーティライブラリ管理 |
| 静的解析 | ruff（lint + format） |

---

## ディレクトリ構成

```
uv260305_phyvac/
├── doc/
│   └── implementation_plan.md   # 本ドキュメント
├── src/
│   └── phyvac/
│       ├── __init__.py           # 公開APIの再エクスポート
│       ├── psychrometrics.py     # 湿り空気計算         ✅ 完了
│       ├── heat_exchanger.py     # 熱交換器基本計算     ✅ 完了
│       ├── components.py         # 各種空調機器クラス   ✅ 完了
│       ├── control.py            # 制御クラス           ⬜ 未着手
│       └── systems.py            # 配管・流量計算クラス  ⬜ 未着手
├── tests/
│   ├── test_psychrometrics.py    ⬜ 未着手
│   ├── test_heat_exchanger.py    ⬜ 未着手
│   ├── test_components.py        ⬜ 未着手
│   ├── test_control.py           ⬜ 未着手
│   └── test_systems.py           ⬜ 未着手
├── equipment_spec.xlsx           # 機器仕様表（既存）
├── phyvac.py                     # 元のモノリシックファイル（参照用）
├── pyproject.toml
└── uv.lock
```

---

## モジュール分割方針

### 分割の基本原則

| 原則 | 説明 |
|------|------|
| **単一責任** | 各モジュールは1つの関心事（物理計算 / 機器モデル / 制御 / 流体系）のみ扱う |
| **依存方向の統一** | 上位モジュールが下位モジュールに依存する一方向の依存グラフを維持する |
| **循環参照の排除** | モジュール間に循環 import が生じない構成とする |
| **後方互換性** | `from phyvac import Pump` など既存の使い方が `__init__.py` を介して引き続き動作する |

### モジュール依存グラフ

```
systems.py
    │  依存
    ▼
components.py ──依存──► heat_exchanger.py ──依存──► psychrometrics.py
    │                                                        ▲
control.py ─────────────────────────────────────────────────┘（間接依存）
```

- `psychrometrics.py`: 外部依存なし（標準ライブラリ `math` のみ）
- `heat_exchanger.py`: `psychrometrics` に依存
- `components.py`: `psychrometrics`, `heat_exchanger` に依存
- `control.py`: 外部依存なし（`numpy` のみ）。phyvac 内部モジュールに依存しない
- `systems.py`: `components.py` のクラスをコンポジションで受け取るが、import レベルでは依存しない

> **Note**: `systems.py` はコンポジションパターンにより、`Pump`・`Fan`・`Valve`・`Damper` 等のオブジェクトを引数として受け取る。型注釈目的以外では `components` を import しない設計とする。

### モジュール別 import 構造

#### `psychrometrics.py`
```python
import math
from scipy import optimize
```

#### `heat_exchanger.py`
```python
import math
from scipy import optimize
from .psychrometrics import tdb2hsat
```

#### `components.py`
```python
import math
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

from .heat_exchanger import getparameter_hex, hex_effectiveness, hex_ntu
from .psychrometrics import (
    h_rh2w, tdb2hsat, tdb_rh2h_x, tdb_rh2twb, tdb_twb2w,
    tdb_w2h, tdp2psat, pv2w, w2cpair, w_h2tdb, w_rh2tdb, w_tdb2rh,
)
```

#### `control.py`
```python
import numpy as np
```

#### `systems.py`
```python
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .components import quadratic_formula  # quadratic_formula のみ import

if TYPE_CHECKING:
    from .components import Pump, Fan, Valve, Damper, Pump_para
```

> `quadratic_formula` は元コードでは `phyvac.py` の関数として定義されており `Pump_para`, `BranchW` 等から呼ばれる。`Pump.figure_curve()` および `Fan.figure_curve()` からも呼ばれるため、`components.py` に配置し `systems.py` から `from .components import quadratic_formula` でインポートする。

#### `__init__.py`
```python
# 公開APIを一箇所から再エクスポート
from .psychrometrics import *
from .heat_exchanger import *
from .components import *
from .control import *
from .systems import *
```

---

## 各モジュールの詳細

### `psychrometrics.py`（完了）

**役割**: 湿り空気の熱力学的状態量計算。純粋関数のみ、副作用なし。

**設計方針**:
- すべて純粋関数（ステートレス）
- 引数: スカラー値（`float`）のみ。NumPy配列は受け付けない
- 物理定数はモジュール先頭に定義（`CA`, `CV`, `R0` など）

| 関数 | 入力 | 出力 | 依存 |
|------|------|------|------|
| `tdb2psat` | tdb | psat [kPa] | — |
| `tdb_rh2tdp` | tdb, rh | tdp [°C] | `tdb2psat` |
| `tdb_rh2h_x` | tdb, rh | h [kJ/kg'], x [kg/kg'] | `tdb2psat` |
| `tdb_rh2twb` | tdb, rh | twb [°C] | scipy.optimize |
| `tdb_w2h` | tdb, w | h [kJ/kg'] | — |
| `tdb2hsat` | tdb | hsat [kJ/kg'] | `tdb2psat` |
| `w2pv` | w | pv [kPa] | — |
| `pv2w` | pv | w [kg/kg'] | — |
| `tdp2psat` | tdp | psat [kPa] | — |
| `h_rh2w` | h, rh | w [kg/kg'] | scipy.optimize |
| `tdb2den` | tdb | ρ [kg/m³] | — |
| `h_rh2tdb` | h, rh | tdb [°C] | `h_rh2w` |
| `tdb_rh2h` | tdb, rh | h [kJ/kg'] | `tdb_rh2h_x` |
| `tdb_rh2w` | tdb, rh | w [kg/kg'] | `tdb2psat` |
| `psat2tdp` | psat | tdp [°C] | — |
| `w_h2tdb` | w, h | tdb [°C] | — |
| `w_rh2tdb` | w, rh | tdb [°C] | `w2pv` |
| `w2cpair` | w | cp [kJ/kg'·K] | — |
| `w_tdb2rh` | w, tdb | rh [%] | `tdb2psat` |
| `tdb_twb2w` | tdb, twb | w [kg/kg'] | `tdb_rh2twb`, scipy.optimize |

---

### `heat_exchanger.py`（完了）

**役割**: NTU-効率法（ε-NTU法）に基づく熱交換器の基本計算。純粋関数のみ。

**設計方針**:
- 向流・並列流の2種類に対応
- 無効な `flowtype` は `ValueError` を送出（元コードの `print` 文を改善）
- `hex_ntu` は `scipy.optimize.newton` による数値逆算

| 関数 | 入力 | 出力 |
|------|------|------|
| `getparameter_hex` | tdb | (fa, fb) — 飽和比エンタルピーの線形近似係数 |
| `hex_effectiveness` | ntu, ratio_heat_cap, flowtype | eff [-] |
| `hex_ntu` | feff, fratio_heat_cap, fflowtype | NTU [-] |

---

### `components.py`（完了）

**役割**: 空調・熱源機器のシミュレーションクラス群。

**設計方針**:
- 各クラスは機器1台分の状態（流量・温度・消費電力 等）をインスタンス変数として保持
- `cal()` メソッドで1タイムステップ分の計算を実行し、インスタンス変数を更新する
- `f2p(g)`: 流量 → 圧力差（配管計算との連携用）
- `p2f(dp)`: 圧力差 → 流量（同上）
- `f2p_co()`: `f2p` の二次近似係数を返す（`BranchW` 等の係数連立方程式に使用）

**クラス一覧と依存ライブラリ**:

| クラス | Excel依存 | 外部ライブラリ | 主なメソッド |
|--------|-----------|----------------|--------------|
| `Valve` | なし | numpy | `f2p`, `p2f`, `f2p_co` |
| `Pump` | なし | numpy, matplotlib | `f2p`, `f2p_co`, `cal`, `figure_curve` |
| `Chiller` | あり | pandas, scipy(interpolate) | `cal` |
| `AirSourceHeatPump` | あり | pandas, scipy(interpolate) | `cal` |
| `AbsorptionChillerESS` | なし | — | `cal_c`, `cal_h` |
| `VariableRefrigerantFlowESS` | なし | — | `cal_c`, `cal_h` |
| `VariableRefrigerantFlowEP` | あり | pandas, sklearn | `cal`, `cal_pl`, `cal_loss` |
| `VRFEPHeatingMode` | あり | pandas, sklearn | `cal`, `cal_pl`, `cal_loss` |
| `GeoThermalHeatPump_LCEM` | なし | — | `run`, `get_config`, `set_config` |
| `CoolingTower` | なし | — | `cal`, `f2p`, `f2p_co` |
| `AHU_simple` | なし | — | `cal`, `f2p`, `f2p_co` |
| `VerticalWaterThermalStorageTank` | なし | numpy | `cal`, `TDMA_solver` |
| `HeatExchangerW2W` | なし | — | `cal` |
| `HeatExchangerW2A` | なし | scipy(optimize) | `cal` |
| `Damper` | なし | — | `f2p`, `p2f` |
| `Fan` | なし | numpy, matplotlib | `f2p`, `f2p_co`, `cal`, `figure_curve` |
| `SteamSprayHumidifier` | なし | — | `cal` |

**メソッド規約の統一**:

```
f2p(g) -> float        # 流量[m³/min] → 圧力差[kPa or Pa]
p2f(dp) -> float       # 圧力差 → 流量
f2p_co() -> list       # f2p の二次多項式係数 [a0, a1, a2]
cal(...) -> ...        # 1ステップ計算（副作用あり、self.* を更新）
```

---

### `control.py`（未完了）

**役割**: フィードバック制御・台数制御ロジック。

**設計方針**:
- **コンポジションパターン**: `PumpWithBypassValve`, `BypassValve` は `PID` オブジェクトを `__init__` で受け取る
- `PID.control(sp, mv)`: 設定値(sp)と計測値(mv)から操作量(a)を1ステップ更新して返す
- 積分リセット: 偏差が一定時間同符号のとき `sig` をリセットし、ワインドアップを防止
- 台数制御は「効果待ち時間」を `flag_switch` 配列（シフトレジスタ）で実装

| クラス | コンポジション元 | 主なメソッド |
|--------|----------------|--------------|
| `PID` | — | `control(sp, mv) -> float` |
| `PumpWithBypassValve` | `PID`×2 | `control(sp, mv) -> [float, float]` |
| `BypassValve` | `PID`×1 | `control(sp, mv, thre) -> float` |
| `UnitNum` | — | `control(g) -> int` |
| `UnitNumChiller` | — | `control(g, q) -> int` |

---

### `systems.py`（未完了）

**役割**: 配管・ダクト系の流量-圧力連立方程式。コンポジションパターンで機器オブジェクトを受け取り、系統全体の流量を計算する。

**設計方針**:
- `BranchW` / `BranchA` が系統計算の基本単位（1本の枝）
- `Pump_para` は複数台の並列ポンプ+バイパス弁を1台のポンプと同等インタフェースで扱う
- 二次方程式の解法は `quadratic_formula` に集約し、各クラスから呼び出す
- `f2p_co()` で返す係数 `[a0, a1, a2]` は `dp = a0 + a1*g + a2*g²` の形式で統一

**`BranchW` の分岐ロジック**:

```
BranchW
├── pump=None, valve=None  → 配管のみ（圧損 = kr_eq + kr_pipe）
├── pump=None, valve=あり  → 二方弁あり（弁圧損 + 配管圧損）
├── pump=あり, valve=None  → ポンプあり（揚程 - 配管圧損）
│       ├── pump.para == 0 → 単体ポンプ（Pump）
│       └── pump.para == 1 → 並列ポンプ（Pump_para）
└── pump=あり, valve=あり  → ポンプ + 二方弁（揚程 + 弁 - 配管圧損）
        ├── pump.para == 0 → 単体ポンプ
        └── pump.para == 1 → 並列ポンプ
```

**`Pump_para` のバイパス弁連立方程式**:

```
ポンプ側:  dp = pg(g_pump) - kr_pipe_pump * g_pump²
弁側:      dp = vlv(g_valve) - kr_pipe_valve * g_valve²
質量保存:  num * g_pump - g_valve = g_total
```

3式を g_pump の二次方程式に整理し `quadratic_formula` で解く。

| クラス/関数 | 接続可能なオブジェクト | 主なメソッド |
|------------|----------------------|--------------|
| `quadratic_formula` | — | `(g, flag)` を返す |
| `Pump_para` | `Pump`, `Valve` | `f2p`, `p2f`, `f2p_co` |
| `BranchW` | `Pump`/`Pump_para`, `Valve` | `f2p`, `p2f` |
| `BranchW1` | `Pump`, `Valve`（バイパス） | `p2f` |
| `BranchA` | `Fan`, `Damper` | `f2p`, `p2f` |
| `RoomSimple` | — | `cal(q, gin, tin, rhin, cal_interval)` |

---

## 命名規則

### 変数名

| 接頭辞/接尾辞 | 意味 | 例 |
|--------------|------|----|
| `tdb` | 乾球温度 [°C] | `tdb_in`, `tout_ch` |
| `twb` | 湿球温度 [°C] | `twb_d` |
| `tdp` | 露点温度 [°C] | `tdp_room` |
| `rh` | 相対湿度 [%] | `rhin_air` |
| `w` | 絶対湿度 [kg/kg'] | `w_in_air` |
| `h` | 比エンタルピー [kJ/kg'] | `hin_air` |
| `g` | 体積流量 [m³/min] | `g_ch`, `g_water` |
| `dp` | 圧力差 [kPa] (水系) / [Pa] (空気系) | `dp_pump` |
| `pw` | 消費電力 [kW] | `pw_d` |
| `q` | 熱量 [kW] | `q_ch` |
| `inv` | インバータ周波数比 [-] (0~1) | `inv` |
| `_d` | 定格値 (rated) | `g_ch_d`, `pw_d` |
| `_sp` | 設定値 (setpoint) | `tout_ch_sp` |
| `kr` | 圧損係数 [kPa/(m³/min)²] | `kr_eq`, `kr_pipe` |
| `ef` | 効率 [-] | `ef`, `r_ef` |
| `cop` | 成績係数 [-] | `cop`, `COP_rp` |
| `pl` | 部分負荷率 [-] | `pl`, `plr` |

### メソッド名

| 名前 | 意味 |
|------|------|
| `f2p(g)` | flow to pressure（流量 → 圧力差） |
| `p2f(dp)` | pressure to flow（圧力差 → 流量） |
| `f2p_co()` | f2p の多項式係数を返す |
| `cal(...)` | 1タイムステップ計算（状態更新） |
| `control(...)` | 制御演算（1タイムステップ） |

---

## 既知のバグと修正方針

元の`phyvac.py`に存在するバグを修正しながらリファクタリングする。

| # | 場所 | バグ内容 | 修正方針 | 状態 |
|---|------|----------|----------|------|
| 1 | `SteamSprayHumidifier.cal()` L2842-2843 | `psy_psat_tsat()`, `psy_w_pv()` が未定義 | `tdp2psat()`, `pv2w()` に置換（psychrometrics から import） | ✅ 修正済み |
| 2 | `HeatExchangerW2A.__init__()` L2418 | `hex_ntu()` を引数4個で呼び出し（正しくは3個） | `hex_ntu(eff, cap_min/cap_max, 'counterflow')` に修正 | ✅ 修正済み |
| 3 | `GeoThermalHeatPump_LCEM.set_config()` L1603 | `rated_freq` の `self.` 抜け | `self.rated_freq` に修正 | ✅ 修正済み |
| 4 | `AbsorptionChillerESS.cal_c()` L798 | `tout_chsp` が未定義（正しくはローカル変数 `tout_ch_sp`） | `tout_ch_sp + delta_t` に修正（`self.` は付けない） | ✅ 修正済み |

---

## テスト方針

```
tests/
├── test_psychrometrics.py    # 各関数の境界値・物理的整合性テスト
├── test_heat_exchanger.py    # NTU・有効度の整合性テスト
├── test_components.py        # 各クラスのcal()・f2p()・p2f()テスト
├── test_control.py           # PID制御・台数制御の動作テスト
└── test_systems.py           # 配管枝の流量-圧力計算テスト
```

### テスト対象の優先度

| 優先度 | 対象 | 方針 |
|--------|------|------|
| 高 | `psychrometrics`, `heat_exchanger` | 全関数をテスト。物理的整合性（`rh=100` → 露点=乾球、NTU→∞ → eff→1 など） |
| 中 | `Valve`, `Pump`, `CoolingTower`, `Damper`, `Fan`, `SteamSprayHumidifier`, `AbsorptionChillerESS`, `GeoThermalHeatPump_LCEM`, `AHU_simple`, `VerticalWaterThermalStorageTank`, `HeatExchangerW2W`, `HeatExchangerW2A` | Excelファイル不要クラス。デフォルト引数でのインスタンス化・`cal()`呼び出し |
| 中 | `PID`, `PumpWithBypassValve`, `BypassValve`, `UnitNum`, `UnitNumChiller` | 制御ステップを複数回実行し、収束・切替動作を確認 |
| 中 | `quadratic_formula`, `BranchW`, `BranchA`, `Pump_para` | 典型的な配管構成での流量計算 |
| 低 | `Chiller`, `AirSourceHeatPump`, `VariableRefrigerantFlowEP`, `VRFEPHeatingMode` | Excel依存。`pytest.mark.skip` または `unittest.mock.patch` でファイルI/Oを回避 |

### 物理的整合性のテスト例

```python
# psychrometrics
def test_saturation_at_rh100():
    """rh=100のとき露点温度=乾球温度。"""
    tdb = 20.0
    assert tdb_rh2tdp(tdb, 100.0) == pytest.approx(tdb, abs=0.01)

# heat_exchanger
def test_hex_effectiveness_limits():
    """NTU→∞ のとき有効度→1（向流、ratio<1）。"""
    eff = hex_effectiveness(1e6, 0.5, "counterflow")
    assert eff == pytest.approx(1.0, abs=1e-4)

def test_hex_ntu_roundtrip():
    """hex_ntu(hex_effectiveness(ntu)) == ntu。"""
    ntu = 2.5
    ratio = 0.7
    eff = hex_effectiveness(ntu, ratio, "counterflow")
    assert hex_ntu(eff, ratio, "counterflow") == pytest.approx(ntu, rel=1e-5)
```

---

## pyproject.toml 更新方針

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/phyvac"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]  # 長い行は docstring 内に存在するため

[tool.ruff.lint.isort]
known-first-party = ["phyvac"]
```

---

## 実装ステップ

| # | タスク | 状態 |
|---|--------|------|
| 1 | `src/phyvac/psychrometrics.py` 作成 | ✅ 完了 |
| 2 | `src/phyvac/heat_exchanger.py` 作成 | ✅ 完了 |
| 3 | `src/phyvac/components.py` 作成（バグ修正込み） | ✅ 完了 |
| 4 | `src/phyvac/control.py` 作成 | ⬜ 未着手 |
| 5 | `src/phyvac/systems.py` 作成 | ⬜ 未着手 |
| 6 | `src/phyvac/__init__.py` 作成 | ⬜ 未着手 |
| 7 | `tests/test_psychrometrics.py` 作成 | ⬜ 未着手 |
| 8 | `tests/test_heat_exchanger.py` 作成 | ⬜ 未着手 |
| 9 | `tests/test_components.py` 作成 | ⬜ 未着手 |
| 10 | `tests/test_control.py` 作成 | ⬜ 未着手 |
| 11 | `tests/test_systems.py` 作成 | ⬜ 未着手 |
| 12 | `pyproject.toml` 更新 | ⬜ 未着手 |

---

## 依存ライブラリ

### ランタイム依存（既に `pyproject.toml` に記載済み）

| ライブラリ | 用途 | 使用モジュール |
|-----------|------|----------------|
| `numpy` | 配列演算（BranchW/A、PID 等） | components, control, systems |
| `scipy` | `optimize.newton/brentq`（収束計算）、`interpolate.RegularGridInterpolator` | psychrometrics, heat_exchanger, components |
| `pandas` | Excelスペックシート読み込み | components（Chiller, VRF 等） |
| `matplotlib` | Pump/Fanの性能曲線表示 | components |
| `scikit-learn` | VRFの回帰分析（LinearRegression） | components |

### 開発依存（既に `pyproject.toml` に記載済み）

| ライブラリ | 用途 |
|-----------|------|
| `pytest` | ユニットテスト |
| `pytest-cov` | カバレッジ計測 |
| `ruff` | lint + format |
| `ty` | 型チェック |
| `marimo` | インタラクティブノートブック |

---

## 設計上の注意点

### 1. `Pump`/`Fan` のインスタンス名自動取得

`traceback.extract_stack()` でインスタンス化した行のテキストからインスタンス名を取得する仕組みがある。

```python
(filename, line_number, function_name, text) = traceback.extract_stack()[-2]
self.name = text[:text.find('=')].strip()
```

この処理は `src` パッケージへの移行後も機能するが、テスト時は `pump = Pump(...)` となるため `self.name = "pump"` になる。グラフタイトルに使われるだけなので機能上の問題はない。

### 2. `figure=1` デフォルトによる `plt.show()` 呼び出し

`Pump.__init__` と `Fan.__init__` は `figure=1` がデフォルトで、インスタンス生成時に `figure_curve()` → `plt.show()` が呼ばれる。テストでは必ず `figure=0` を指定する。

```python
# テストでの正しい書き方
pump = Pump(figure=0)
fan = Fan(figure=0)
```

### 3. Excelファイル依存クラスのテスト

`Chiller`, `AirSourceHeatPump`, `VariableRefrigerantFlowEP`, `VRFEPHeatingMode` は `__init__` で `pd.read_excel('equipment_spec.xlsx', ...)` を実行する。

テスト戦略:
- **Option A**: `pytest.mark.skip(reason="requires equipment_spec.xlsx")`
- **Option B**: `unittest.mock.patch('pandas.read_excel', ...)` でモック化

### 4. コンポジションパターンの型注釈

`BranchW`, `BranchA`, `Pump_para` は duck typing で設計されており、`pump` 引数に `Pump` または `Pump_para` が受け入れられる。型注釈は `Union` または `Protocol` で表現する。

```python
from typing import Union, Protocol

class PumpLike(Protocol):
    inv: float
    g: float
    para: int
    def f2p(self, g: float) -> float: ...
    def f2p_co(self) -> list[float]: ...
```

### 5. `AbsorptionChillerESS.cal_c()` の typo（修正済み）

元コード L798 に `tout_chsp` という typo がある。`tout_ch_sp` は `cal_c()` の**ローカル変数**であり、インスタンス属性ではない点に注意。

```python
# 元コード（バグ）
self.tout_ch = tout_chsp + delta_t   # NameError: tout_chsp は未定義

# 修正後（components.py 適用済み）
self.tout_ch = tout_ch_sp + delta_t  # ローカル変数 tout_ch_sp を参照
# ※ self.tout_ch_sp とすると AttributeError になるため self. は付けない
```
