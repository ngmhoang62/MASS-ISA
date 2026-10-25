from __future__ import annotations

import json
import random
from functools import partial

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5TokenizerFast, get_linear_schedule_with_warmup

from .data import ABSADataset, parse_xml_to_data
from .evaluation import append_epoch_metrics, evaluate, save_final_predictions
from .logging_utils import log_message
from .models import build_model


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int, base_seed: int):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(config, tokenizer):
    train_data = parse_xml_to_data(config.train_xml_file, config)
    test_data = parse_xml_to_data(config.test_xml_file, config)
    if not train_data:
        raise ValueError(f"No training samples were loaded from {config.train_xml_file}")
    if not test_data:
        raise ValueError(f"No test samples were loaded from {config.test_xml_file}")
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    worker_init_fn = partial(_seed_worker, base_seed=config.seed)
    train_loader = DataLoader(
        ABSADataset(train_data, tokenizer, config),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        ABSADataset(test_data, tokenizer, config),
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader


def build_optimizer(model, config):
    if config.optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")
    optimizer_groups = [
        {"params": model.backbone_parameters(), "lr": config.backbone_lr},
        {"params": model.head_parameters(), "lr": config.head_lr},
    ]
    optimizer_kwargs = {"weight_decay": config.weight_decay}
    if config.use_fused_adam and config.device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        return AdamW(optimizer_groups, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        return AdamW(optimizer_groups, **optimizer_kwargs)


def build_scheduler(optimizer, config, total_steps):
    if config.scheduler_name != "linear":
        raise ValueError(f"Unsupported scheduler: {config.scheduler_name}")
    warmup_steps = int(config.warmup_ratio * total_steps)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def save_checkpoint(model, config):
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if config.checkpoint_trainable_only:
        state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, config.checkpoint_path)


def load_checkpoint(model, checkpoint_path, config):
    try:
        state_dict = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=config.device)
    is_legacy_notebook_checkpoint = any(key.startswith("flan_t5.") for key in state_dict)
    if is_legacy_notebook_checkpoint:
        if config.use_lora:
            raise ValueError("Legacy notebook checkpoints are compatible only with the Base full architecture")
        remapped = {}
        for key, value in state_dict.items():
            if key == "flan_t5.shared.weight":
                remapped["encoder.shared.weight"] = value
            elif key.startswith("flan_t5.encoder."):
                remapped["encoder.encoder." + key[len("flan_t5.encoder.") :]] = value
            elif not key.startswith("flan_t5."):
                remapped[key] = value
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Legacy checkpoint conversion failed; missing={missing}, unexpected={unexpected}"
            )
        return
    model.load_state_dict(state_dict, strict=not config.checkpoint_trainable_only)


def save_best_results(metrics, config):
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload["resolved_config"] = config.as_serializable_dict()
    with config.results_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    log_message(config, "RESULT", f"Saved evaluation summary to {config.results_path}")


def _validate_config(config):
    if config.epochs < 1:
        raise ValueError("training.epochs must be at least 1")
    if config.batch_size < 1 or config.evaluation_batch_size < 1:
        raise ValueError("training batch sizes must be at least 1")
    if not config.train_xml_file.exists():
        raise FileNotFoundError(f"Training XML file not found: {config.train_xml_file}")
    if not config.test_xml_file.exists():
        raise FileNotFoundError(f"Test XML file not found: {config.test_xml_file}")
    allowed_metrics = {"all_accuracy", "all_macro_f1", "isa_accuracy", "isa_macro_f1"}
    if config.checkpoint_selection_metric not in allowed_metrics:
        raise ValueError(
            f"training.checkpoint_selection_metric must be one of {sorted(allowed_metrics)}"
        )


def train(config):
    _validate_config(config)
    set_seed(config.seed, config.deterministic)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    if config.epoch_metrics_path.exists():
        config.epoch_metrics_path.unlink()

    log_message(config, "CONFIG", f"Dataset: {config.dataset}")
    log_message(config, "CONFIG", f"Variant: MASS-{config.variant.upper()}")
    log_message(config, "CONFIG", f"Ablation: {config.ablation}")
    log_message(config, "CONFIG", f"Seed: {config.seed}")
    log_message(config, "CONFIG", f"Train file: {config.train_xml_file}")
    log_message(config, "CONFIG", f"Test file: {config.test_xml_file}")
    log_message(config, "CONFIG", f"Device: {config.device}")
    log_message(config, "SETUP", "Loading tokenizer and datasets.")

    tokenizer = T5TokenizerFast.from_pretrained(config.model_name)
    train_loader, test_loader = create_dataloaders(config, tokenizer)
    log_message(config, "DATA", f"Train samples: {len(train_loader.dataset)}")
    log_message(config, "DATA", f"Test samples: {len(test_loader.dataset)}")

    log_message(config, "SETUP", "Loading model.")
    model = build_model(config).to(config.device)
    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * config.epochs
    scheduler = build_scheduler(optimizer, config, total_steps)

    best_score = float("-inf")
    best_metrics = None
    log_message(config, "TRAIN", "Starting training.")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_subj_loss = 0.0
        epoch_pol_loss = 0.0
        epoch_ortho_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["input_ids"].to(config.device, non_blocking=config.pin_memory),
                batch["attention_mask"].to(config.device, non_blocking=config.pin_memory),
                batch["representation_mask"].to(config.device, non_blocking=config.pin_memory),
                batch["numeric_label"].to(config.device, non_blocking=config.pin_memory),
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            if outputs["loss_subj"] is not None:
                epoch_subj_loss += outputs["loss_subj"].item()
                epoch_pol_loss += outputs["loss_pol"].item()
                epoch_ortho_loss += outputs["loss_ortho"].item()

        denominator = max(len(train_loader), 1)
        train_metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss / denominator,
            "subjectivity_loss": epoch_subj_loss / denominator,
            "polarity_loss": epoch_pol_loss / denominator,
            "orthogonality_loss": epoch_ortho_loss / denominator,
        }
        log_message(
            config,
            "TRAIN",
            f"Epoch {epoch}/{config.epochs} | Loss={train_metrics['train_loss']:.6f}",
        )

        evaluation_metrics = evaluate(model, test_loader, config, epoch)
        append_epoch_metrics(config.epoch_metrics_path, {**train_metrics, **evaluation_metrics})
        selection_score = evaluation_metrics[config.checkpoint_selection_metric]
        if selection_score > best_score:
            best_score = selection_score
            save_checkpoint(model, config)
            best_metrics = {
                "dataset": config.dataset,
                "variant": config.variant,
                "ablation": config.ablation,
                "seed": config.seed,
                "best_epoch": epoch,
                **evaluation_metrics,
                "checkpoint_path": str(config.checkpoint_path),
                "predictions_path": str(config.predictions_path),
                "isa_predictions_path": str(config.isa_predictions_path),
            }
            log_message(
                config,
                "CHECKPOINT",
                f"Updated best model at epoch {epoch} | "
                f"{config.checkpoint_selection_metric}={selection_score * 100:.2f}%",
            )

    if best_metrics is None:
        raise RuntimeError("Training finished without a checkpoint")

    log_message(config, "RESULT", "Loading the selected checkpoint and writing predictions.")
    load_checkpoint(model, config.checkpoint_path, config)
    final_metrics = save_final_predictions(model, test_loader, config)
    best_metrics.update(final_metrics)
    save_best_results(best_metrics, config)
    return best_metrics
