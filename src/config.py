from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class MassConfig:
    """Fully resolved settings for one MASS run."""

    config_path: Path
    project_root: Path
    variant: str
    dataset: str
    seed: int
    ablation: str
    use_aspect_marker: bool
    representation_mode: str
    use_two_subspaces: bool
    use_orthogonality: bool
    use_class_weights: bool
    train_xml_file: Path
    test_xml_file: Path
    model_name: str
    model_dtype: str
    output_dir: Path
    log_dir: Path
    checkpoint_path: Path
    predictions_path: Path
    isa_predictions_path: Path
    results_path: Path
    epoch_metrics_path: Path
    aggregate_results_path: Path
    max_source_length: int
    truncation_strategy: str
    start_marker: str
    end_marker: str
    batch_size: int
    evaluation_batch_size: int
    epochs: int
    backbone_lr: float
    head_lr: float
    proj_dim: int
    projection_dropout: float
    ortho_weight: float
    label_smoothing: float
    subj_loss_weights: Tuple[float, float]
    polarity_loss_weights: Tuple[float, float]
    optimizer_name: str
    weight_decay: float
    scheduler_name: str
    warmup_ratio: float
    gradient_clip_norm: float
    checkpoint_selection_metric: str
    conditional_polarity_inference: bool
    device: torch.device
    deterministic: bool
    num_workers: int
    pin_memory: bool
    use_fused_adam: bool
    checkpoint_trainable_only: bool
    use_lora: bool
    lora_rank: int | None
    lora_alpha: int | None
    lora_dropout: float | None
    lora_target_modules: Tuple[str, ...]

    def as_serializable_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        if not self.use_lora:
            for key in (
                "use_lora",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_target_modules",
            ):
                values.pop(key)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
            elif isinstance(value, torch.device):
                values[key] = str(value)
        return values


def load_yaml(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return raw


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing config key '{context}.{key}'")
    return mapping[key]


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _format_path(base_dir: Path, template: str, values: Mapping[str, Any]) -> Path:
    return _resolve_path(base_dir, template.format(**values))


def _resolve_device(value: str) -> torch.device:
    normalized = value.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config.yaml requests CUDA, but CUDA is not available")
    return device


def _tuple_of_floats(values: Iterable[Any], expected_length: int, name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")
    return result


def get_run_defaults(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    raw = load_yaml(config_path)
    return dict(_require(raw, "run", "root"))


def get_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    variant: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    ablation: str | None = None,
) -> MassConfig:
    """Resolve config.yaml into one validated experiment configuration."""

    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    raw = load_yaml(config_path)
    run = _require(raw, "run", "root")

    variant_name = (variant or _require(run, "default_variant", "run")).lower()
    dataset_name = (dataset or _require(run, "default_dataset", "run")).lower()
    seed_value = int(seed if seed is not None else _require(run, "default_seed", "run"))
    ablation_name = (ablation or run.get("default_ablation", "full")).lower()

    models = _require(raw, "models", "root")
    data = _require(raw, "data", "root")
    datasets = _require(data, "datasets", "data")
    ablations = raw.get("ablations", {})
    if variant_name not in models:
        raise ValueError(f"Unsupported MASS variant: {variant_name}")
    if dataset_name not in datasets:
        raise ValueError(f"Unsupported MASS dataset: {dataset_name}")
    if ablation_name not in ablations:
        raise ValueError(
            f"Unsupported MASS ablation '{ablation_name}'. Must be one of {sorted(ablations)}"
        )

    model = models[variant_name]
    dataset_config = datasets[dataset_name]
    ablation_config = ablations[ablation_name]
    architecture = _require(raw, "architecture", "root")
    training = _require(raw, "training", "training") if "training" not in raw else _require(raw, "training", "root")
    runtime = _require(raw, "runtime", "root")
    paths = _require(raw, "paths", "root")
    lora = model.get("lora")
    if lora is not None and not isinstance(lora, dict):
        raise ValueError(f"models.{variant_name}.lora must be a mapping")

    use_aspect_marker = bool(_require(ablation_config, "use_aspect_marker", f"ablations.{ablation_name}"))
    representation_mode = str(_require(ablation_config, "representation_mode", f"ablations.{ablation_name}")).lower()
    allowed_rep_modes = {"start_marker", "end_marker", "mean_span", "mean_prompt"}
    if representation_mode not in allowed_rep_modes:
        raise ValueError(f"Unsupported representation_mode: {representation_mode}")
    use_two_subspaces = bool(_require(ablation_config, "use_two_subspaces", f"ablations.{ablation_name}"))
    use_orthogonality = bool(_require(ablation_config, "use_orthogonality", f"ablations.{ablation_name}"))
    use_class_weights = bool(_require(ablation_config, "use_class_weights", f"ablations.{ablation_name}"))

    path_values = {
        "variant": variant_name,
        "dataset": dataset_name,
        "seed": seed_value,
        "ablation": ablation_name,
    }
    output_root_key = "output_root" if ablation_name == "full" else "ablation_output_root"
    aggregate_key = "aggregate_file" if ablation_name == "full" else "ablation_aggregate_file"
    output_dir = _format_path(project_root, _require(paths, output_root_key, "paths"), path_values)
    use_lora = lora is not None

    return MassConfig(
        config_path=config_path,
        project_root=project_root,
        variant=variant_name,
        dataset=dataset_name,
        seed=seed_value,
        ablation=ablation_name,
        use_aspect_marker=use_aspect_marker,
        representation_mode=representation_mode,
        use_two_subspaces=use_two_subspaces,
        use_orthogonality=use_orthogonality,
        use_class_weights=use_class_weights,
        train_xml_file=_resolve_path(project_root, _require(dataset_config, "train", f"data.datasets.{dataset_name}")),
        test_xml_file=_resolve_path(project_root, _require(dataset_config, "test", f"data.datasets.{dataset_name}")),
        model_name=str(_require(model, "pretrained_name", f"models.{variant_name}")),
        model_dtype=str(_require(model, "dtype", f"models.{variant_name}")),
        output_dir=output_dir,
        log_dir=output_dir / "logs",
        checkpoint_path=output_dir / _require(paths, "checkpoint_file", "paths"),
        predictions_path=output_dir / _require(paths, "predictions_file", "paths"),
        isa_predictions_path=output_dir / _require(paths, "isa_predictions_file", "paths"),
        results_path=output_dir / _require(paths, "metrics_file", "paths"),
        epoch_metrics_path=output_dir / _require(paths, "epoch_metrics_file", "paths"),
        aggregate_results_path=_format_path(project_root, _require(paths, aggregate_key, "paths"), path_values),
        max_source_length=int(_require(data, "max_source_length", "data")),
        truncation_strategy=str(_require(data, "truncation_strategy", "data")),
        start_marker=str(_require(data, "start_marker", "data")),
        end_marker=str(_require(data, "end_marker", "data")),
        batch_size=int(_require(training, "batch_size", "training")),
        evaluation_batch_size=int(_require(training, "evaluation_batch_size", "training")),
        epochs=int(_require(training, "epochs", "training")),
        backbone_lr=float(_require(training, "backbone_learning_rate", "training")),
        head_lr=float(_require(training, "head_learning_rate", "training")),
        proj_dim=int(_require(architecture, "projection_dimension", "architecture")),
        projection_dropout=float(_require(architecture, "projection_dropout", "architecture")),
        ortho_weight=float(_require(training, "orthogonality_weight", "training")),
        label_smoothing=float(_require(training, "label_smoothing", "training")),
        subj_loss_weights=_tuple_of_floats(
            _require(dataset_config, "subjectivity_class_weights", f"data.datasets.{dataset_name}"),
            2,
            "subjectivity_class_weights",
        ),
        polarity_loss_weights=_tuple_of_floats(
            _require(dataset_config, "polarity_class_weights", f"data.datasets.{dataset_name}"),
            2,
            "polarity_class_weights",
        ),
        optimizer_name=str(_require(training, "optimizer", "training")).lower(),
        weight_decay=float(_require(training, "weight_decay", "training")),
        scheduler_name=str(_require(training, "scheduler", "training")).lower(),
        warmup_ratio=float(_require(training, "warmup_ratio", "training")),
        gradient_clip_norm=float(_require(training, "gradient_clip_norm", "training")),
        checkpoint_selection_metric=str(_require(training, "checkpoint_selection_metric", "training")),
        conditional_polarity_inference=bool(
            _require(architecture, "conditional_polarity_inference", "architecture")
        ),
        device=_resolve_device(str(_require(runtime, "device", "runtime"))),
        deterministic=bool(_require(runtime, "deterministic", "runtime")),
        num_workers=int(_require(runtime, "num_workers", "runtime")),
        pin_memory=bool(_require(runtime, "pin_memory", "runtime")),
        use_fused_adam=bool(_require(runtime, "fused_adam", "runtime")),
        checkpoint_trainable_only=bool(
            use_lora and _require(runtime, "checkpoint_trainable_only_for_lora", "runtime")
        ),
        use_lora=use_lora,
        lora_rank=int(_require(lora, "rank", f"models.{variant_name}.lora")) if lora else None,
        lora_alpha=int(_require(lora, "alpha", f"models.{variant_name}.lora")) if lora else None,
        lora_dropout=float(_require(lora, "dropout", f"models.{variant_name}.lora")) if lora else None,
        lora_target_modules=tuple(
            str(item) for item in _require(lora, "target_modules", f"models.{variant_name}.lora")
        ) if lora else (),
    )
