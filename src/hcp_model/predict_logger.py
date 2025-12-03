from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _default_logs_path() -> Path:
    """
    Determine a default logs directory under the project root.

    Project root is assumed to be two levels up from this file:
    src/hcp_model/predict_logger.py -> src -> <project_root>
    """
    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "predictions.log"


@dataclass
class PredictionLogRecord:
    timestamp: str
    booking_id: Optional[str]
    cancellation_probability: float
    predicted_label: int
    risk_bucket: str
    model_version: Optional[str] = None
    source: str = "api"  # 'api', 'batch', etc. if we extend later


class PredictionLogger:
    """
    Append structured prediction events to a JSONL log file.
    """

    def __init__(self, logfile: Optional[Path] = None) -> None:
        if logfile is None:
            logfile = _default_logs_path()
        self.logfile = logfile

    def log_prediction(
        self,
        *,
        booking_id: Optional[str],
        cancellation_probability: float,
        predicted_label: int,
        risk_bucket: str,
        model_version: Optional[str] = None,
        source: str = "api",
    ) -> None:
        """
        Best-effort logging: failures are logged as warnings but
        must not break the calling code.
        """
        record = PredictionLogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            booking_id=booking_id,
            cancellation_probability=float(cancellation_probability),
            predicted_label=int(predicted_label),
            risk_bucket=str(risk_bucket),
            model_version=model_version,
            source=source,
        )

        try:
            with self.logfile.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to log prediction event", exc_info=True)