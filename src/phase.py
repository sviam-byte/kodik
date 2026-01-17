import numpy as np
import pandas as pd

def phase_indicators(df_hist: pd.DataFrame) -> dict:
    if df_hist is None or df_hist.empty:
        return {"fc_step": None, "jump": np.nan, "fc_removed_frac": np.nan}

    if "lcc_frac" not in df_hist.columns:
        return {"fc_step": None, "jump": np.nan, "fc_removed_frac": np.nan}

    s = df_hist["lcc_frac"].to_numpy(dtype=float)
    if len(s) < 2:
        return {"fc_step": int(df_hist["step"].iloc[-1]), "jump": 0.0, "fc_removed_frac": np.nan}

    drops = s[:-1] - s[1:]
    j = float(np.nanmax(drops))
    k = int(np.nanargmax(drops))
    fc_step = int(df_hist["step"].iloc[k])

    if "nodes_left" in df_hist.columns:
        n0 = float(df_hist["nodes_left"].iloc[0])
        n_fc = float(df_hist["nodes_left"].iloc[k])
        fc_removed = 1.0 - (n_fc / max(1.0, n0))
    else:
        fc_removed = np.nan

    return {"fc_step": fc_step, "jump": j, "fc_removed_frac": float(fc_removed)}
