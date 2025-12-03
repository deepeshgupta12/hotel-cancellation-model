from dataclasses import dataclass
from typing import Literal, Optional

RiskBucket = Literal["low", "medium", "high"]


@dataclass
class RiskConfig:
    """
    Configuration for mapping cancellation probabilities to risk buckets.

    low_max:    upper bound for 'low' risk (exclusive)
    medium_max: upper bound for 'medium' risk (exclusive)
    'high' is everything >= medium_max
    """
    low_max: float = 0.2
    medium_max: float = 0.5

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RiskConfig":
        if not d:
            return cls()
        return cls(
            low_max=d.get("low_max", 0.2),
            medium_max=d.get("medium_max", 0.5),
        )


def bucket_from_proba(
    p_cancel: float,
    cfg: Optional[RiskConfig] = None,
) -> RiskBucket:
    """
    Map a cancellation probability (0–1) into a risk bucket.

    Defaults:
      - p < low_max             -> 'low'
      - low_max <= p < medium_max -> 'medium'
      - p >= medium_max         -> 'high'
    """
    if cfg is None:
        cfg = RiskConfig()

    if p_cancel < cfg.low_max:
        return "low"
    if p_cancel < cfg.medium_max:
        return "medium"
    return "high"