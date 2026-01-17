import json
import base64
import pandas as pd


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
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


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
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


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
