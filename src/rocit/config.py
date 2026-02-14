"""YAML configuration loading and validation for ROCIT CLI commands."""

import dataclasses
import yaml
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


class ConfigError(Exception):
    """Raised when a YAML config file is invalid."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration for ``rocit train``."""
    sample_id: str
    labelled_data: str
    sample_distribution: str
    cell_atlas: str
    val_chromosomes: list[str]
    test_chromosomes: list[str]
    output_dir: str
    cache_dir: str = "/scratch"


@dataclass
class PredictConfig:
    """Configuration for ``rocit predict``."""
    sample_id: str
    best_checkpoint_path: str
    sample_distribution: str
    cell_atlas: str
    output_dir: str
    read_store: Optional[str] = None
    read_store_dir: Optional[str] = None
    cache_dir: str = "/scratch"


@dataclass
class PreprocessConfig:
    """Configuration for ``rocit preprocess``."""
    sample_id: str
    bam: str
    methylation_dir: str
    copy_number: str
    variants: str
    haplotags: str
    haploblocks: str
    snv_clusters: str
    output_dir: str
    snv_cluster_assignments: Optional[str] = None


@dataclass
class RunConfig:
    """Configuration for ``rocit run`` (full pipeline)."""
    sample_id: str
    bam: str
    copy_number: str
    variants: str
    haplotags: str
    haploblocks: str
    snv_clusters: str
    cell_atlas: str
    val_chromosomes: list[str]
    test_chromosomes: list[str]
    output_dir: str
    bam_index: Optional[str] = None
    snv_cluster_assignments: Optional[str] = None
    min_mapq: int = 0
    workers: int = 1
    chromosomes: Optional[list[str]] = None
    cache_dir: str = "/scratch"


def load_yaml(config_path: Path) -> dict:
    """Read and parse a YAML file, returning the raw dict."""
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    with open(config_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Failed to parse YAML file {config_path}: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {config_path} must be a YAML mapping "
            f"(got {type(data).__name__})."
        )
    return data


def _required_field_names(config_cls) -> set[str]:
    """Return the set of field names that have no default value."""
    required = set()
    for f in fields(config_cls):
        has_default = f.default is not dataclasses.MISSING
        has_factory = f.default_factory is not dataclasses.MISSING
        if not has_default and not has_factory:
            required.add(f.name)
    return required


def validate_config(config_cls, data: dict, config_path: Path):
    """Instantiate a config dataclass from *data* with helpful errors."""
    cls_fields = {f.name for f in fields(config_cls)}
    required = _required_field_names(config_cls)
    provided = set(data.keys())

    unknown = provided - cls_fields
    if unknown:
        raise ConfigError(
            f"Unknown keys in {config_path}: {sorted(unknown)}. "
            f"Valid keys are: {sorted(cls_fields)}"
        )

    missing = required - provided
    if missing:
        raise ConfigError(
            f"Missing required keys in {config_path}: {sorted(missing)}"
        )

    try:
        return config_cls(**data)
    except TypeError as exc:
        raise ConfigError(
            f"Invalid config in {config_path}: {exc}"
        ) from exc


def load_config(config_cls, config_path: Path):
    """Top-level entry: load YAML, validate, return typed config."""
    data = load_yaml(config_path)
    return validate_config(config_cls, data, config_path)



def resolve_path(path_str: str, must_exist: bool = True) -> Path:
    """Convert a string from YAML to a validated Path."""
    p = Path(path_str).expanduser().resolve()
    if must_exist and not p.exists():
        raise ConfigError(
            f"Path does not exist: {p} (from config value '{path_str}')"
        )
    return p


def resolve_file(path_str: str) -> Path:
    """Resolve and verify the path points to an existing file."""
    p = resolve_path(path_str, must_exist=True)
    if not p.is_file():
        raise ConfigError(f"Expected a file but got: {p}")
    return p


def resolve_dir(path_str: str, must_exist: bool = True) -> Path:
    """Resolve and verify the path points to a directory."""
    p = resolve_path(path_str, must_exist=must_exist)
    if must_exist and not p.is_dir():
        raise ConfigError(f"Expected a directory but got: {p}")
    return p


def validate_train_config(cfg: TrainConfig) -> None:
    """Validate cross-field constraints for training config."""
    if not cfg.val_chromosomes:
        raise ConfigError("val_chromosomes must not be empty.")
    if not cfg.test_chromosomes:
        raise ConfigError("test_chromosomes must not be empty.")

    overlap = set(cfg.val_chromosomes) & set(cfg.test_chromosomes)
    if overlap:
        raise ConfigError(
            f"val_chromosomes and test_chromosomes overlap: {sorted(overlap)}"
        )


def validate_predict_config(cfg: PredictConfig) -> None:
    """Validate mutual exclusion for predict config."""
    if cfg.read_store and cfg.read_store_dir:
        raise ConfigError(
            "read_store and read_store_dir are mutually exclusive. "
            "Provide only one."
        )
    if not cfg.read_store and not cfg.read_store_dir:
        raise ConfigError(
            "You must provide either read_store or read_store_dir."
        )


def validate_run_config(cfg: RunConfig) -> None:
    """Validate cross-field constraints for the full pipeline config."""
    if not cfg.val_chromosomes:
        raise ConfigError("val_chromosomes must not be empty.")
    if not cfg.test_chromosomes:
        raise ConfigError("test_chromosomes must not be empty.")

    overlap = set(cfg.val_chromosomes) & set(cfg.test_chromosomes)
    if overlap:
        raise ConfigError(
            f"val_chromosomes and test_chromosomes overlap: {sorted(overlap)}"
        )
