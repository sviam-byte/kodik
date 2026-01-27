import csv
import io
from typing import Optional, Tuple

import pandas as pd


def _sniff_csv_dialect(file_bytes: bytes, encoding: str) -> Tuple[str, bool]:
    """Try to detect a delimiter/header from a small decoded prefix."""
    prefix = file_bytes[: 64 * 1024]
    text = prefix.decode(encoding, errors="strict")
    candidates = [",", ";", "\t", "|"]

    try:
        dialect = csv.Sniffer().sniff(text, delimiters=candidates)
        sep = dialect.delimiter
    except Exception:
        lines = text.splitlines()[:50]
        counts = {d: sum(line.count(d) for line in lines) for d in candidates}
        sep = max(counts, key=counts.get) if counts else ","

    try:
        has_header = csv.Sniffer().has_header(text)
    except Exception:
        has_header = True

    return sep, has_header


def _read_csv_fast_with_encoding_fallback(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV bytes quickly with pragmatic encoding fallbacks."""
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ]

    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            sep, _has_header = _sniff_csv_dialect(file_bytes, enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue

        bio = io.BytesIO(file_bytes)
        try:
            return pd.read_csv(bio, sep=sep, engine="c", encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue

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
        # Fast path: try comma-separated first (fast C parser).
        # If delimiter is wrong (common with ';' in connectome CSVs), the file collapses into 1 column.
        bio = io.BytesIO(file_bytes)
        try:
            df = pd.read_csv(bio, engine="c", low_memory=True)
            # Heuristic: wrong delimiter -> single-column dataframe.
            if df.shape[1] <= 1:
                df = _read_csv_fast_with_encoding_fallback(file_bytes)
        except UnicodeDecodeError:
            df = _read_csv_fast_with_encoding_fallback(file_bytes)
        except Exception:
            # ParserError / quoting issues / etc.
            df = _read_csv_fast_with_encoding_fallback(file_bytes)
        except Exception:
            # ParserError / quoting issues / etc.
            df = _read_csv_fast_with_encoding_fallback(file_bytes)
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
