import io
import pandas as pd


def load_uploaded_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load uploaded bytes into a DataFrame for CSV or Excel inputs."""
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        df = pd.read_csv(bio, sep=None, engine="python")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(bio)
    else:
        raise ValueError("Неподдерживаемый формат файла (нужен csv/xlsx/xls)")

    df.columns = [str(c).strip() for c in df.columns]
    return df
