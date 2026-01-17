import json
import base64
import pandas as pd
import numpy as np


def _json_safe(x):
    """
    Recursively convert common numpy/pandas types to JSON-serializable python objects.
    This prevents: TypeError: Object of type ... is not JSON serializable
    """
    if x is None or isinstance(x, (str, bool, int, float)):
        return x

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if not isinstance(k, str):
                k = str(k)
            out[k] = _json_safe(v)
        return out

    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    if isinstance(x, set):
        return [_json_safe(v) for v in sorted(list(x), key=lambda z: str(z))]

    if isinstance(x, pd.Series):
        return _json_safe(x.to_list())
    if isinstance(x, pd.DataFrame):
        return _json_safe(x.to_dict(orient="records"))

    return str(x)


def _json_dumps_bytes(payload: dict) -> bytes:
    safe_payload = _json_safe(payload)
    return json.dumps(safe_payload, ensure_ascii=False, indent=2).encode("utf-8")


def _df_to_b64_csv(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to base64-encoded CSV bytes."""
    csv = df.to_csv(index=False).encode("utf-8")
    return base64.b64encode(csv).decode("ascii")


def _b64_csv_to_df(s: str) -> pd.DataFrame:
    """Deserialize a base64-encoded CSV string into a DataFrame."""
    raw = base64.b64decode(s.encode("ascii"))
    return pd.read_csv(pd.io.common.BytesIO(raw))


def export_workspace_json(graphs: dict, experiments: list) -> bytes:
    """
    graphs: dict[gid] -> {id,name,source,tags,edges(df),created_at}
    experiments: list -> {id,name,graph_id,attack_kind,params,history(df),created_at}
    """
    g_out = {}
    for gid, g in graphs.items():
        g_out[gid] = {
            "id": g["id"],
            "name": g["name"],
            "source": g["source"],
            "tags": g.get("tags", {}),
            "created_at": g.get("created_at", 0.0),
            "edges_b64": _df_to_b64_csv(g["edges"]),
        }

    e_out = []
    for e in experiments:
        e_out.append(
            {
                "id": e["id"],
                "name": e["name"],
                "graph_id": e["graph_id"],
                "attack_kind": e["attack_kind"],
                "params": e.get("params", {}),
                "created_at": e.get("created_at", 0.0),
                "history_b64": _df_to_b64_csv(e["history"]),
            }
        )

    payload = {"graphs": g_out, "experiments": e_out}
    return _json_dumps_bytes(payload)


def import_workspace_json(blob: bytes) -> tuple[dict, list]:
    """Load workspace graphs and experiments from a JSON blob."""
    payload = json.loads(blob.decode("utf-8"))
    graphs_raw = payload.get("graphs", {})
    exps_raw = payload.get("experiments", [])

    graphs = {}
    for gid, g in graphs_raw.items():
        edges = _b64_csv_to_df(g["edges_b64"])
        graphs[gid] = {
            "id": g.get("id", gid),
            "name": g.get("name", gid),
            "source": g.get("source", "import"),
            "tags": g.get("tags", {}),
            "created_at": g.get("created_at", 0.0),
            "edges": edges,
        }

    exps = []
    for e in exps_raw:
        hist = _b64_csv_to_df(e["history_b64"])
        exps.append(
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "graph_id": e.get("graph_id"),
                "attack_kind": e.get("attack_kind"),
                "params": e.get("params", {}),
                "created_at": e.get("created_at", 0.0),
                "history": hist,
            }
        )

    return graphs, exps


def export_experiments_json(experiments: list) -> bytes:
    """Export experiments only (without graph storage) as JSON bytes."""
    e_out = []
    for e in experiments:
        e_out.append(
            {
                "id": e["id"],
                "name": e["name"],
                "graph_id": e["graph_id"],
                "attack_kind": e["attack_kind"],
                "params": e.get("params", {}),
                "created_at": e.get("created_at", 0.0),
                "history_b64": _df_to_b64_csv(e["history"]),
            }
        )
    payload = {"experiments": e_out}
    return _json_dumps_bytes(payload)


def import_experiments_json(blob: bytes) -> list:
    """Import experiments from JSON bytes."""
    payload = json.loads(blob.decode("utf-8"))
    exps_raw = payload.get("experiments", [])
    exps = []
    for e in exps_raw:
        hist = _b64_csv_to_df(e["history_b64"])
        exps.append(
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "graph_id": e.get("graph_id"),
                "attack_kind": e.get("attack_kind"),
                "params": e.get("params", {}),
                "created_at": e.get("created_at", 0.0),
                "history": hist,
            }
        )
    return exps
