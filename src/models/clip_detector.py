"""CLIP vision backbone plus a minimal single-logit classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


class FineTuneMode(StrEnum):
    HEAD_ONLY = "head_only"
    LAST_BLOCK = "last_block"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class DetectorOutput:
    logits: Tensor
    embeddings: Tensor | None = None


class CLIPVisionBackbone(nn.Module):
    """Wrap ``transformers.CLIPVisionModel`` and expose pooled embeddings."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        feature_source: str = "pooled_output",
        encoder: nn.Module | None = None,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if feature_source not in {"pooled_output", "cls_token"}:
            raise ValueError("feature_source must be pooled_output or cls_token")
        if encoder is None:
            try:
                from transformers import CLIPVisionModel
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required to load CLIP; install project dependencies"
                ) from exc
            if revision is None:
                encoder = CLIPVisionModel.from_pretrained(model_name)
            else:
                encoder = CLIPVisionModel.from_pretrained(model_name, revision=revision)
        self.encoder = encoder
        self.model_name = model_name
        self.revision = revision
        self.feature_source = feature_source
        inferred = getattr(getattr(encoder, "config", None), "hidden_size", None)
        resolved_dim = feature_dim if feature_dim is not None else inferred
        if not isinstance(resolved_dim, int) or resolved_dim <= 0:
            raise ValueError("could not determine a positive CLIP feature dimension")
        self.feature_dim: int = resolved_dim

    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape [B, 3, H, W]")
        if not pixel_values.is_floating_point() or not torch.isfinite(pixel_values).all():
            raise ValueError("pixel_values must be finite floating-point values")
        output: Any = self.encoder(pixel_values=pixel_values)
        if self.feature_source == "pooled_output":
            embeddings = getattr(output, "pooler_output", None)
            if embeddings is None and isinstance(output, dict):
                embeddings = output.get("pooler_output")
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, dict):
                hidden = output.get("last_hidden_state")
            embeddings = None if hidden is None else hidden[:, 0, :]
        if not isinstance(embeddings, Tensor):
            raise RuntimeError(f"CLIP output does not provide {self.feature_source}")
        if embeddings.ndim != 2 or embeddings.shape != (pixel_values.shape[0], self.feature_dim):
            raise RuntimeError(
                f"backbone returned {tuple(embeddings.shape)}; expected "
                f"[{pixel_values.shape[0]}, {self.feature_dim}]"
            )
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("backbone produced non-finite embeddings")
        return embeddings


class BinaryClassifierHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_dim:
            raise ValueError(f"embeddings must have shape [B, {self.input_dim}]")
        return cast(Tensor, self.classifier(self.dropout(embeddings)).squeeze(-1))


class CLIPBinaryDetector(nn.Module):
    def __init__(self, backbone: CLIPVisionBackbone, classifier: BinaryClassifierHead) -> None:
        super().__init__()
        if backbone.feature_dim != classifier.input_dim:
            raise ValueError("classifier input dimension does not match backbone output")
        self.backbone = backbone
        self.classifier = classifier
        self.trainability_summary: dict[str, Any] = {}

    def forward(self, pixel_values: Tensor, *, return_embeddings: bool = False) -> DetectorOutput:
        embeddings = self.backbone(pixel_values)
        logits = self.classifier(embeddings)
        if logits.ndim != 1 or logits.shape[0] != pixel_values.shape[0]:
            raise RuntimeError("detector must return one logit per image")
        return DetectorOutput(logits=logits, embeddings=embeddings if return_embeddings else None)

    def predict_proba(self, pixel_values: Tensor) -> Tensor:
        return torch.sigmoid(self(pixel_values).logits)

    def predict(self, pixel_values: Tensor, *, threshold: float = 0.5) -> Tensor:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(pixel_values) >= threshold).to(torch.int64)


def _last_vision_block(backbone: CLIPVisionBackbone) -> nn.Module:
    encoder = backbone.encoder
    candidates: list[Any] = [
        getattr(getattr(getattr(encoder, "vision_model", None), "encoder", None), "layers", None),
        getattr(encoder, "layers", None),
    ]
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return cast(nn.Module, layers[-1])
    raise ValueError("cannot locate the final vision transformer block")


def configure_trainable_layers(model: CLIPBinaryDetector, mode: FineTuneMode | str) -> None:
    try:
        selected_mode = FineTuneMode(mode)
    except ValueError as exc:
        raise ValueError(f"unknown fine-tune mode: {mode}") from exc
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if selected_mode == FineTuneMode.LAST_BLOCK:
        for parameter in _last_vision_block(model.backbone).parameters():
            parameter.requires_grad = True
        post_norm = getattr(
            getattr(model.backbone.encoder, "vision_model", None), "post_layernorm", None
        )
        if post_norm is not None:
            for parameter in post_norm.parameters():
                parameter.requires_grad = True
    elif selected_mode == FineTuneMode.FULL:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model.trainability_summary = {
        "mode": selected_mode.value,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_names": trainable_names,
    }
    LOGGER.info("fine-tune mode=%s trainable=%d/%d", selected_mode.value, trainable, total)
