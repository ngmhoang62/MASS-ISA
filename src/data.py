from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .labels import POLARITY_TO_NUM


def build_prompt(
    *,
    term: str,
    text: str,
    start_marker: str,
    end_marker: str,
    use_aspect_marker: bool = True,
) -> Tuple[str, str]:
    if use_aspect_marker:
        marked_text = text.replace(
            term,
            f"{start_marker} {term} {end_marker}",
        )
    else:
        marked_text = text
    prompt = (
        f"Task: Analyze sentiment for the aspect [{term}]. "
        "Step 1: Is the context stating an objective fact (Neutral) or expressing a "
        "subjective opinion (Polar)? "
        "Step 2: If subjective, is it Positive or Negative? "
        f"Context: {marked_text}"
    )
    return prompt, marked_text


def parse_xml_to_data(xml_path: Path | str, config) -> List[Dict[str, Any]]:
    """Parse one SemEval XML split and remove conflict-labeled instances."""

    tree = ET.parse(xml_path)
    root = tree.getroot()
    data: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(root.findall("sentence")):
        text_node = sentence.find("text")
        aspect_terms_node = sentence.find("aspectTerms")
        if text_node is None or text_node.text is None or aspect_terms_node is None:
            continue

        text = text_node.text
        sentence_id = sentence.get("id", str(sentence_index))
        for aspect_index, aspect in enumerate(aspect_terms_node.findall("aspectTerm")):
            term = aspect.get("term")
            polarity = aspect.get("polarity")
            if not term or polarity not in POLARITY_TO_NUM:
                continue

            prompt, marked_text = build_prompt(
                term=term,
                text=text,
                start_marker=config.start_marker,
                end_marker=config.end_marker,
                use_aspect_marker=config.use_aspect_marker,
            )

            data.append(
                {
                    "sample_id": f"{sentence_id}:{aspect_index}",
                    "sentence_id": sentence_id,
                    "prompt": prompt,
                    "marked_text": marked_text,
                    "prompt_context_from": prompt.rfind(marked_text),
                    "numeric_label": POLARITY_TO_NUM[polarity],
                    "is_implicit": str(aspect.get("implicit_sentiment", "False")).lower() == "true",
                    "raw_text": text,
                    "raw_aspect": term,
                }
            )

    return data


class ABSADataset(Dataset):
    """Tokenized ABSA data with the start-marker representation used by MASS."""

    def __init__(self, data, tokenizer, config):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("MASS requires a fast T5 tokenizer")
        if config.truncation_strategy != "target_centered":
            raise ValueError(
                f"Unsupported data.truncation_strategy: {config.truncation_strategy}"
            )
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.start_marker_id = tokenizer.convert_tokens_to_ids(config.start_marker)
        self.end_marker_id = tokenizer.convert_tokens_to_ids(config.end_marker)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prefix = item["prompt"][: item["prompt_context_from"]]
        prefix_ids = self.tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        context_ids = self.tokenizer(
            item["marked_text"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        context_budget = self.config.max_source_length - len(prefix_ids) - 1
        start_marker_position = None
        end_marker_position = None

        if self.config.use_aspect_marker:
            start_positions = (context_ids == self.start_marker_id).nonzero(as_tuple=True)[0].tolist()
            end_positions = (context_ids == self.end_marker_id).nonzero(as_tuple=True)[0].tolist()
            if not start_positions or not end_positions:
                raise ValueError(f"Aspect markers are missing for sample {item['sample_id']}")
            target_start, target_end = start_positions[0], end_positions[0] + 1
            if target_end <= target_start + 1:
                raise ValueError(f"Invalid marker span for sample {item['sample_id']}")

            target_length = target_end - target_start
            if context_budget < target_length:
                raise ValueError(
                    f"The prompt leaves insufficient room for sample {item['sample_id']} at "
                    f"data.max_source_length={self.config.max_source_length}"
                )
            if len(context_ids) > context_budget:
                remaining = context_budget - target_length
                window_start = max(0, target_start - (remaining // 2))
                window_end = min(len(context_ids), window_start + context_budget)
                window_start = max(0, window_end - context_budget)
            else:
                window_start, window_end = 0, len(context_ids)

            context_ids = context_ids[window_start:window_end]
            start_marker_position = len(prefix_ids) + target_start - window_start
            end_marker_position = len(prefix_ids) + (target_end - 1) - window_start
        else:
            if context_budget < 1:
                raise ValueError(
                    f"The prompt leaves insufficient room for sample {item['sample_id']} at "
                    f"data.max_source_length={self.config.max_source_length}"
                )
            context_ids = context_ids[:context_budget]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("The configured T5 tokenizer does not define an EOS token")

        input_ids = torch.cat(
            [prefix_ids, context_ids, torch.tensor([eos_id], dtype=torch.long)]
        )
        attention_mask = torch.ones_like(input_ids)
        padding_length = self.config.max_source_length - len(input_ids)
        if padding_length < 0:
            raise AssertionError("Target-preserving truncation exceeded max_source_length")
        if padding_length:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long),
                ]
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(padding_length, dtype=torch.long)]
            )

        representation_mask = torch.zeros(self.config.max_source_length, dtype=torch.bool)
        if self.config.representation_mode == "start_marker":
            representation_mask[start_marker_position] = True
        elif self.config.representation_mode == "end_marker":
            representation_mask[end_marker_position] = True
        elif self.config.representation_mode == "mean_span":
            representation_mask[start_marker_position : end_marker_position + 1] = True
        elif self.config.representation_mode == "mean_prompt":
            representation_mask = attention_mask.bool()
        else:
            raise ValueError(f"Unsupported representation_mode: {self.config.representation_mode}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "representation_mask": representation_mask,
            "numeric_label": torch.tensor(item["numeric_label"], dtype=torch.long),
            "is_implicit": torch.tensor(item["is_implicit"], dtype=torch.bool),
            "sample_id": item["sample_id"],
            "sentence_id": item["sentence_id"],
            "raw_text": item["raw_text"],
            "raw_aspect": item["raw_aspect"],
        }
