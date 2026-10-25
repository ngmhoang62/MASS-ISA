from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel


def _torch_dtype(name: str):
    normalized = name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported model dtype: {name}")


class MassModel(nn.Module):
    """Encoder-only implementation of the MASS hierarchy."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder(config)
        hidden_dim = self.encoder.config.d_model

        if config.use_two_subspaces:
            self.proj_subj = self._make_projection(
                hidden_dim,
                config.proj_dim,
                config.projection_dropout,
            )
            self.proj_pol = self._make_projection(
                hidden_dim,
                config.proj_dim,
                config.projection_dropout,
            )
            self.classifier_subj = nn.Linear(config.proj_dim, 2)
            self.classifier_pol = nn.Linear(config.proj_dim, 2)

            subj_weights = (
                torch.tensor(config.subj_loss_weights, dtype=torch.float32)
                if config.use_class_weights
                else None
            )
            pol_weights = (
                torch.tensor(config.polarity_loss_weights, dtype=torch.float32)
                if config.use_class_weights
                else None
            )
            self.loss_fn_subj = nn.CrossEntropyLoss(
                weight=subj_weights,
                label_smoothing=config.label_smoothing,
            )
            self.loss_fn_pol = nn.CrossEntropyLoss(
                weight=pol_weights,
                label_smoothing=config.label_smoothing,
                reduction="none",
            )
        else:
            self.proj_main = self._make_projection(
                hidden_dim,
                config.proj_dim,
                config.projection_dropout,
            )
            self.classifier_main = nn.Linear(config.proj_dim, 3)
            self.loss_fn_main = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing,
            )

    @staticmethod
    def _build_encoder(config):
        encoder = T5EncoderModel.from_pretrained(
            config.model_name,
            torch_dtype=_torch_dtype(config.model_dtype),
        )
        if not config.use_lora:
            return encoder

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("The XXL variant requires the 'peft' package from requirements.txt") from exc

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
        )
        encoder = get_peft_model(encoder, peft_config)
        encoder.print_trainable_parameters()
        return encoder

    @staticmethod
    def _make_projection(hidden_dim: int, proj_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    @staticmethod
    def _pool_hidden_states(hidden_states: torch.Tensor, representation_mask: torch.Tensor) -> torch.Tensor:
        mask = representation_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return ((hidden_states * mask).sum(dim=1) / denominator).float()

    def forward(
        self,
        input_ids,
        attention_mask,
        representation_mask,
        labels=None,
        *,
        conditional_polarity=False,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        aspect_states = self._pool_hidden_states(
            encoder_outputs.last_hidden_state,
            representation_mask,
        )
        if not self.config.use_two_subspaces:
            z_main = self.proj_main(aspect_states)
            logits_main = self.classifier_main(z_main).float()
            result = {
                "loss": None,
                "loss_subj": None,
                "loss_pol": None,
                "loss_ortho": None,
                "logits_subj": None,
                "logits_pol": None,
                "logits_main": logits_main,
            }
            if labels is not None:
                result["loss"] = self.loss_fn_main(logits_main, labels)
            return result

        z_subj = self.proj_subj(aspect_states)
        logits_subj = self.classifier_subj(z_subj).float()

        if labels is None and conditional_polarity:
            logits_pol = torch.zeros(
                input_ids.shape[0],
                2,
                device=logits_subj.device,
                dtype=logits_subj.dtype,
            )
            polar_mask = torch.argmax(logits_subj, dim=1) == 1
            if polar_mask.any():
                logits_pol[polar_mask] = self.classifier_pol(
                    self.proj_pol(aspect_states[polar_mask])
                ).float()
        else:
            logits_pol = self.classifier_pol(self.proj_pol(aspect_states)).float()

        result = {
            "loss": None,
            "loss_subj": None,
            "loss_pol": None,
            "loss_ortho": None,
            "logits_subj": logits_subj,
            "logits_pol": logits_pol,
            "logits_main": None,
        }
        if labels is None:
            return result

        subj_labels = (labels != 2).long()
        loss_subj = self.loss_fn_subj(logits_subj, subj_labels)
        polar_mask = labels != 2
        if polar_mask.any():
            raw_loss_pol = self.loss_fn_pol(logits_pol[polar_mask], labels[polar_mask])
            loss_pol = raw_loss_pol.mean()
        else:
            loss_pol = logits_pol.sum() * 0.0

        if self.config.use_orthogonality and self.config.ortho_weight > 0.0:
            weight_subj = F.normalize(self.proj_subj[0].weight.float(), p=2, dim=1)
            weight_pol = F.normalize(self.proj_pol[0].weight.float(), p=2, dim=1)
            loss_ortho = torch.linalg.matrix_norm(weight_subj @ weight_pol.T, ord="fro")
            total_loss = loss_subj + loss_pol + (self.config.ortho_weight * loss_ortho)
        else:
            loss_ortho = logits_subj.sum() * 0.0
            total_loss = loss_subj + loss_pol

        result["loss_subj"] = loss_subj
        result["loss_pol"] = loss_pol
        result["loss_ortho"] = loss_ortho
        result["loss"] = total_loss
        return result

    def backbone_parameters(self):
        if self.config.use_lora:
            return [parameter for parameter in self.encoder.parameters() if parameter.requires_grad]
        return list(self.encoder.parameters())

    def head_parameters(self):
        if not self.config.use_two_subspaces:
            modules = [self.proj_main, self.classifier_main]
        else:
            modules = [
                self.proj_subj,
                self.proj_pol,
                self.classifier_subj,
                self.classifier_pol,
            ]
        return [parameter for module in modules for parameter in module.parameters()]


def build_model(config):
    return MassModel(config)
