from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .labels import NUM_TO_POLARITY
from .logging_utils import log_message


SENTIMENT_LABELS = (0, 1, 2)


def decode_outputs(outputs) -> List[int]:
    if outputs.get("logits_main") is not None:
        return torch.argmax(outputs["logits_main"], dim=1).cpu().tolist()
    pred_subj = torch.argmax(outputs["logits_subj"], dim=1)
    pred_pol = torch.argmax(outputs["logits_pol"], dim=1)
    return [2 if pred_subj[index].item() == 0 else pred_pol[index].item() for index in range(len(pred_subj))]


def _move_batch(batch, config):
    return (
        batch["input_ids"].to(config.device, non_blocking=config.pin_memory),
        batch["attention_mask"].to(config.device, non_blocking=config.pin_memory),
        batch["representation_mask"].to(config.device, non_blocking=config.pin_memory),
    )


def collect_prediction_records(model, dataloader, config, *, show_progress=False):
    model.eval()
    records = []
    iterator = tqdm(dataloader, desc="Inference") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            input_ids, attention_mask, representation_mask = _move_batch(batch, config)
            outputs = model(
                input_ids,
                attention_mask,
                representation_mask,
                conditional_polarity=config.conditional_polarity_inference,
            )
            predictions = decode_outputs(outputs)
            truths = batch["numeric_label"].cpu().tolist()

            for index, prediction in enumerate(predictions):
                truth = int(truths[index])
                records.append(
                    {
                        "Sample_ID": batch["sample_id"][index],
                        "Sentence_ID": batch["sentence_id"][index],
                        "Sentence": batch["raw_text"][index],
                        "Aspect": batch["raw_aspect"][index],
                        "True_Label": NUM_TO_POLARITY[truth],
                        "Predicted_Label": NUM_TO_POLARITY[prediction],
                        "True_ID": truth,
                        "Predicted_ID": prediction,
                        "Is_Implicit": bool(batch["is_implicit"][index].item()),
                        "Is_Correct": prediction == truth,
                    }
                )
    return records


def metrics_from_records(records: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    truths = np.array([int(record["True_ID"]) for record in records], dtype=np.int64)
    predictions = np.array([int(record["Predicted_ID"]) for record in records], dtype=np.int64)
    implicit_mask = np.array([bool(record["Is_Implicit"]) for record in records], dtype=bool)

    def score(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0, 0.0
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(SENTIMENT_LABELS),
            average="macro",
            zero_division=0,
        )
        return float(accuracy), float(macro_f1)

    all_accuracy, all_macro_f1 = score(truths, predictions)
    isa_accuracy, isa_macro_f1 = score(truths[implicit_mask], predictions[implicit_mask])
    return {
        "all_accuracy": all_accuracy,
        "all_macro_f1": all_macro_f1,
        "isa_accuracy": isa_accuracy,
        "isa_macro_f1": isa_macro_f1,
    }


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def evaluate(model, dataloader, config, epoch):
    records = collect_prediction_records(model, dataloader, config)
    metrics = metrics_from_records(records)
    log_message(
        config,
        "EVAL",
        f"Epoch {epoch}: All | Acc={metrics['all_accuracy'] * 100:.2f}% | "
        f"MF1={metrics['all_macro_f1'] * 100:.2f}%",
    )
    log_message(
        config,
        "EVAL",
        f"Epoch {epoch}: ISA | Acc={metrics['isa_accuracy'] * 100:.2f}% | "
        f"MF1={metrics['isa_macro_f1'] * 100:.2f}%",
    )
    return metrics


def save_final_predictions(model, dataloader, config):
    records = collect_prediction_records(model, dataloader, config, show_progress=True)
    _write_csv(config.predictions_path, records)
    _write_csv(config.isa_predictions_path, [record for record in records if record["Is_Implicit"]])
    log_message(config, "RESULT", f"Saved full-test predictions to {config.predictions_path}")
    log_message(config, "RESULT", f"Saved ISA predictions to {config.isa_predictions_path}")
    return metrics_from_records(records)


def append_epoch_metrics(path: Path, metrics: Mapping[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(metrics), ensure_ascii=False) + "\n")


def aggregate_seed_metrics(run_metrics: Iterable[Mapping[str, object]], output_path: Path):
    metrics_list = list(run_metrics)
    if not metrics_list:
        raise ValueError("No seed metrics were supplied for aggregation")

    metric_names = ("all_accuracy", "all_macro_f1", "isa_accuracy", "isa_macro_f1")
    aggregate = {
        "variant": metrics_list[0]["variant"],
        "dataset": metrics_list[0]["dataset"],
        "ablation": metrics_list[0].get("ablation", "full"),
        "seeds": [int(item["seed"]) for item in metrics_list],
        "num_seeds": len(metrics_list),
        "metrics": {},
        "per_seed": metrics_list,
    }
    for name in metric_names:
        values = np.array([float(item[name]) for item in metrics_list], dtype=np.float64)
        aggregate["metrics"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2)
    return aggregate
