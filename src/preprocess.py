import pandas as pd

def preprocess_fixed_format(df_any: pd.DataFrame):
    if df_any.shape[1] < 10:
        raise ValueError("Expected >= 10 columns: src, dst, ..., confidence(9), weight(10)")

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

    df = df.rename(columns={CONF_COL: "confidence", WEIGHT_COL: "weight"})
    df = df.dropna(subset=[SRC_COL, DST_COL, "confidence", "weight"])
    df = df[df["weight"] > 0]

    return df, SRC_COL, DST_COL
