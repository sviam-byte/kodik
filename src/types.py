"""Typed schemas for structured data passed between modules."""

from typing import TypedDict


class GraphMetrics(TypedDict, total=False):
    """Schema for graph topological metrics."""

    # Basic
    N: int
    E: int
    density: float
    avg_degree: float

    # Components
    C: int
    lcc_size: int
    lcc_frac: float

    # Advanced
    mod: float
    l2_lcc: float
    eff_w: float

    # Entropy & Geometry
    H_deg: float
    H_w: float
    H_conf: float
    kappa_mean: float
    fragility_H: float


class AttackResult(TypedDict):
    """Schema for a single attack history step."""

    step: int
    removed_frac: float
    metrics: GraphMetrics
