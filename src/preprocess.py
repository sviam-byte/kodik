import pandas as pd
import numpy as np


def coerce_fixed_format(df_any: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Ожидается:
    - 1-я колонка source id
    - 2-я колонка target id
    - 9-я confidence
    - 10-я weight
    Возвращает df_edges со столбцами:
      [SRC_COL, DST_COL, weight, confidence]
    и meta: {src_col, dst_col}
    """
    if df_any.shape[1] < 10:
        raise ValueError("Файл должен содержать минимум 10 колонок (фикс. формат).")

    SRC_COL = df_any.columns[0]
    DST_COL = df_any.columns[1]
    CONF_COL = df_any.columns[8]
    WEIGHT_COL = df_any.columns[9]

    df = df_any.copy()

    df[SRC_COL] = pd.to_numeric(df[SRC_COL], errors="coerce").astype("Int64")
    df[DST_COL] = pd.to_numeric(df[DST_COL], errors="coerce").astype("Int64")

    df[CONF_COL] = pd.to_numeric(df[CONF_COL], errors="coerce")

    df[WEIGHT_COL] = pd.to_numeric(
        df[WEIGHT_COL].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # unify columns
    out = df[[SRC_COL, DST_COL, CONF_COL, WEIGHT_COL]].copy()
    out = out.rename(columns={CONF_COL: "confidence", WEIGHT_COL: "weight"})
    out = out.dropna(subset=[SRC_COL, DST_COL, "confidence", "weight"])
    out = out[out["weight"] > 0]

    if out.empty:
        raise ValueError("После очистки данные пустые (проверь numeric confidence/weight и id).")

    meta = {"src_col": SRC_COL, "dst_col": DST_COL}
    return out, meta


def filter_edges(
    df_edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    min_conf: int,
    min_weight: float,
) -> pd.DataFrame:
    """Filter edges by confidence/weight and coerce numeric columns."""
    df = df_edges.copy()
    if "confidence" in df.columns:
        df = df[df["confidence"] >= float(min_conf)]
    if "weight" in df.columns:
        df = df[df["weight"] >= float(min_weight)]

    # ensure clean numeric
    df[src_col] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64")
    df[dst_col] = pd.to_numeric(df[dst_col], errors="coerce").astype("Int64")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    df = df.dropna(subset=[src_col, dst_col, "confidence", "weight"])
    df = df[df["weight"] > 0]
    return df
