import io
import pandas as pd


def _read_csv_with_encoding_fallback(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV bytes with a small set of pragmatic encoding fallbacks.

    Many public datasets (incl. some connectome CSV exports) are encoded as
    Windows-1252/Latin-1 rather than UTF-8, which triggers:
      UnicodeDecodeError: 'utf-8' codec can't decode byte 0x..
    """

    # Order matters: try strict UTF-8 first, then common fallbacks.
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",  # handles BOM
        "cp1252",
        "latin-1",
        "cp1251",
    ]

    last_err: Exception | None = None
    for enc in encodings_to_try:
        bio = io.BytesIO(file_bytes)
        try:
            return pd.read_csv(bio, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
        except Exception:
            # Not a decoding problem (delimiter/malformed/etc.) -> raise immediately.
            raise

    raise UnicodeDecodeError(
        "utf-8",
        file_bytes,
        getattr(last_err, "start", 0),
        getattr(last_err, "end", 0),
        f"Не удалось декодировать CSV. Пробовали кодировки: {', '.join(encodings_to_try)}",
    )


def load_uploaded_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load uploaded bytes into a DataFrame for CSV or Excel inputs."""
    name = (filename or "").lower()

    if name.endswith(".csv"):
        df = _read_csv_with_encoding_fallback(file_bytes)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        bio = io.BytesIO(file_bytes)
        df = pd.read_excel(bio)
    else:
        raise ValueError("Неподдерживаемый формат файла (нужен csv/xlsx/xls)")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_fixed_format(df_any: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean fixed format by position:
      0: src id
      1: dst id
      8: confidence
      9: weight
    """
    if df_any.shape[1] < 10:
        raise ValueError("Need >= 10 columns (fixed format).")

    src_col = df_any.columns[0]
    dst_col = df_any.columns[1]
    conf_col = df_any.columns[8]
    w_col = df_any.columns[9]

    df = df_any.copy()

    # Normalize types and allow for missing/invalid values.
    df[src_col] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64")
    df[dst_col] = pd.to_numeric(df[dst_col], errors="coerce").astype("Int64")
    df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
    df[w_col] = pd.to_numeric(
        df[w_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df = df.rename(columns={conf_col: "confidence", w_col: "weight"})
    df = df.dropna(subset=[src_col, dst_col, "confidence", "weight"])
    df = df[df["weight"] > 0]

    meta = {
        "SRC_COL": src_col,
        "DST_COL": dst_col,
        "CONF_COL": "confidence",
        "WEIGHT_COL": "weight",
    }
    return df, meta
