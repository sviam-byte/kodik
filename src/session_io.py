import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

FORMAT_VERSION = "kodik.workspace.v1"


def _to_builtin(value: Any) -> Any:
    """
    Convert numpy/pandas scalars to JSON-friendly Python primitives.
    """
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Serialize a DataFrame to a list of row dictionaries with built-in types.
    """
    df_clean = df.copy()
    df_clean.columns = [str(c) for c in df_clean.columns]
    records = df_clean.to_dict(orient="records")
    return [{str(k): _to_builtin(v) for k, v in row.items()} for row in records]


def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Deserialize records (list of dicts) into a DataFrame.
    """
    if records is None:
        return pd.DataFrame()
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    return pd.DataFrame.from_records(records)


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    """Dump payload to UTF-8 JSON bytes."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _loads_bytes(blob: bytes) -> Dict[str, Any]:
    """Load a JSON payload from bytes or string-like input."""
    if isinstance(blob, (bytes, bytearray)):
        text = blob.decode("utf-8")
    else:
        text = str(blob)
    return json.loads(text)


def export_workspace_json(graphs: Dict[str, Any], experiments: List[Dict[str, Any]]) -> bytes:
    """
    graphs: session_state["graphs"] (dict)
    experiments: session_state["experiments"] (list)
    Returns bytes suitable for st.download_button.
    """
    ts = time.time()

    graphs_out: Dict[str, Any] = {}
    for gid, g in (graphs or {}).items():
        edges_df = g.get("edges")
        if isinstance(edges_df, pd.DataFrame):
            edges_ser = _df_to_records(edges_df)
        else:
            edges_ser = edges_df if edges_df is not None else []

        graphs_out[str(gid)] = {
            "id": _to_builtin(g.get("id", gid)),
            "name": _to_builtin(g.get("name", "")),
            "source": _to_builtin(g.get("source", "")),
            "tags": g.get("tags", {}) or {},
            "created_at": _to_builtin(g.get("created_at", ts)),
            "edges": edges_ser,
        }

    exps_out: List[Dict[str, Any]] = []
    for e in (experiments or []):
        hist = e.get("history")
        if isinstance(hist, pd.DataFrame):
            hist_ser = _df_to_records(hist)
        else:
            hist_ser = hist if hist is not None else []

        exps_out.append(
            {
                "id": _to_builtin(e.get("id")),
                "name": _to_builtin(e.get("name")),
                "graph_id": _to_builtin(e.get("graph_id")),
                "attack_kind": _to_builtin(e.get("attack_kind")),
                "params": e.get("params", {}) or {},
                "created_at": _to_builtin(e.get("created_at", ts)),
                "history": hist_ser,
            }
        )

    payload = {
        "format": FORMAT_VERSION,
        "exported_at": ts,
        "graphs": graphs_out,
        "experiments": exps_out,
        "meta": {"note": "Kodik Lab workspace export"},
    }
    return _json_bytes(payload)


def import_workspace_json(blob: bytes) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Import workspace graphs and experiments from JSON bytes.
    """
    obj = _loads_bytes(blob)
    graphs_in = obj.get("graphs", {})
    exps_in = obj.get("experiments", [])

    graphs_out: Dict[str, Any] = {}
    for gid, g in (graphs_in or {}).items():
        edges = g.get("edges", [])
        edges_df = _records_to_df(edges) if isinstance(edges, list) else pd.DataFrame()

        graphs_out[str(gid)] = {
            "id": g.get("id", gid),
            "name": g.get("name", ""),
            "source": g.get("source", ""),
            "tags": g.get("tags", {}) or {},
            "created_at": g.get("created_at", time.time()),
            "edges": edges_df,
        }

    exps_out: List[Dict[str, Any]] = []
    for e in (exps_in or []):
        hist = e.get("history", [])
        hist_df = _records_to_df(hist) if isinstance(hist, list) else pd.DataFrame()
        exps_out.append(
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "graph_id": e.get("graph_id"),
                "attack_kind": e.get("attack_kind"),
                "params": e.get("params", {}) or {},
                "created_at": e.get("created_at", time.time()),
                "history": hist_df,
            }
        )

    return graphs_out, exps_out


def export_experiments_json(experiments: List[Dict[str, Any]]) -> bytes:
    """
    Export experiments only (without graph storage) as JSON bytes.
    """
    ts = time.time()
    exps_out: List[Dict[str, Any]] = []
    for e in (experiments or []):
        hist = e.get("history")
        if isinstance(hist, pd.DataFrame):
            hist_ser = _df_to_records(hist)
        else:
            hist_ser = hist if hist is not None else []

        exps_out.append(
            {
                "id": _to_builtin(e.get("id")),
                "name": _to_builtin(e.get("name")),
                "graph_id": _to_builtin(e.get("graph_id")),
                "attack_kind": _to_builtin(e.get("attack_kind")),
                "params": e.get("params", {}) or {},
                "created_at": _to_builtin(e.get("created_at", ts)),
                "history": hist_ser,
            }
        )

    payload = {
        "format": "kodik.experiments.v1",
        "exported_at": ts,
        "experiments": exps_out,
    }
    return _json_bytes(payload)


def import_experiments_json(blob: bytes) -> List[Dict[str, Any]]:
    """
    Import experiments from JSON bytes.
    """
    obj = _loads_bytes(blob)
    exps_in = obj.get("experiments", obj if isinstance(obj, list) else [])
    exps_out: List[Dict[str, Any]] = []
    for e in (exps_in or []):
        hist = e.get("history", [])
        hist_df = _records_to_df(hist) if isinstance(hist, list) else pd.DataFrame()
        exps_out.append(
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "graph_id": e.get("graph_id"),
                "attack_kind": e.get("attack_kind"),
                "params": e.get("params", {}) or {},
                "created_at": e.get("created_at", time.time()),
                "history": hist_df,
            }
        )
    return exps_out
