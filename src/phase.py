import numpy as np
import pandas as pd


def classify_phase_transition(
    df: pd.DataFrame,
    x_col: str = "removed_frac",
    y_col: str = "lcc_frac",
) -> dict:
    """
    Эвристика "взрывного" распада:
    - смотрим на дискретные скачки y между соседними точками
    - если самый большой отрицательный скачок занимает большую долю амплитуды
      и происходит в узком окне x -> считаем abrupt
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return {
            "is_abrupt": False,
            "critical_x": float("nan"),
            "jump": 0.0,
            "jump_fraction": 0.0,
        }

    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)

    if len(x) < 3:
        return {
            "is_abrupt": False,
            "critical_x": float(x[-1]) if len(x) else float("nan"),
            "jump": 0.0,
            "jump_fraction": 0.0,
        }

    dy = np.diff(y)
    # largest negative drop
    idx = int(np.argmin(dy))
    jump = float(-dy[idx])  # positive magnitude of drop
    y_span = float(max(1e-12, np.nanmax(y) - np.nanmin(y)))
    jump_fraction = float(jump / y_span)

    critical_x = float(x[idx + 1]) if idx + 1 < len(x) else float(x[-1])

    # "first-order like" heuristic thresholds
    # jump_fraction >= 0.35 means ~ huge discontinuity relative to range
    is_abrupt = bool(jump_fraction >= 0.35)

    return {
        "is_abrupt": is_abrupt,
        "critical_x": critical_x,
        "jump": jump,
        "jump_fraction": jump_fraction,
    }
