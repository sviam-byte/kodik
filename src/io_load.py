import io
import pandas as pd

def load_uploaded_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        df = pd.read_csv(bio, sep=None, engine="python")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(bio)
    else:
        raise ValueError("Unsupported file type")

    df.columns = [str(c).strip() for c in df.columns]
    return df
