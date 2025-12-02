from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(**raw["paths"])
    training = TrainingConfig(**raw["training"])
    rf_cfg = ModelRFConfig(**raw["model"]["rf"])
    model = ModelConfig(rf=rf_cfg)

    return AppConfig(paths=paths, training=training, model=model)
