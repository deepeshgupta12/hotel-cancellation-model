from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class PathsConfig:
    raw_data: str
    processed_data: str
    scored_data: str
    model_dir: str
    model_file: str


@dataclass
class TrainingConfig:
    test_size: float
    random_state: int
    target_column: str


@dataclass
class ModelRFConfig:
    n_estimators: int
    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int


@dataclass
class ModelConfig:
    rf: ModelRFConfig


@dataclass
class AppConfig:
    paths: PathsConfig
    training: TrainingConfig
    model: ModelConfig
    risk_thresholds: Optional[Dict[str, float]] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    paths_cfg = PathsConfig(**raw["paths"])
    model_cfg = ModelConfig(**raw["model"])
    training_cfg = TrainingConfig(**raw["training"])
    risk_thresholds = raw.get("risk_thresholds")  # may be None if not set

    return AppConfig(
        paths=paths_cfg,
        model=model_cfg,
        training=training_cfg,
        risk_thresholds=risk_thresholds,
    )






    
