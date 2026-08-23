"""Contracts for the cosine (L2-normalised) classifier head.

The cosine head is the model-development arm of the dissertation, so these tests check
the properties the scientific claim depends on rather than merely that the code runs:

* it is **parameter-matched** to the linear head to within one scalar, so a difference in
  results cannot be attributed to extra capacity;
* it is genuinely **invariant to embedding magnitude**, which is the entire hypothesis;
* selecting it is **opt-in**, so every pre-existing config and completed run keeps the
  original linear behaviour untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.clip_detector import (
    BinaryClassifierHead,
    CLIPBinaryDetector,
    CLIPVisionBackbone,
    CosineClassifierHead,
    HeadType,
    build_classifier_head,
    configure_trainable_layers,
)
from src.utils.config import validate_config


class MockEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, *, pixel_values: torch.Tensor) -> object:
        pooled = pixel_values.mean(dim=(2, 3))
        pooled = torch.cat((pooled, pooled[:, :1]), dim=1)
        for layer in self.layers:
            pooled = torch.tanh(layer(pooled))
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=pooled[:, None, :])


# ------------------------------------------------------------------ the core hypothesis


def test_cosine_head_is_invariant_to_embedding_magnitude() -> None:
    """The whole point: rescaling an embedding must not change its logit.

    The linear head's logit scales with the embedding, so it can key on magnitude --
    which tracks generator-specific low-level statistics rather than being evidence of
    generation. The cosine head must be blind to it.
    """

    torch.manual_seed(0)
    embeddings = torch.randn(6, 8)
    cosine = CosineClassifierHead(8).eval()
    linear = BinaryClassifierHead(8).eval()

    with torch.no_grad():
        for factor in (0.1, 2.0, 25.0):
            assert torch.allclose(
                cosine(embeddings), cosine(embeddings * factor), atol=1e-5
            ), f"cosine head changed under scaling by {factor}"
        # The linear head is NOT invariant; if it were, the comparison would be vacuous.
        assert not torch.allclose(linear(embeddings), linear(embeddings * 2.0), atol=1e-5)


def test_cosine_head_still_separates_direction() -> None:
    """Magnitude invariance must not cost the ability to discriminate at all."""

    head = CosineClassifierHead(3).eval()
    with torch.no_grad():
        head.classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        head.classifier.bias.zero_()
        aligned = head(torch.tensor([[5.0, 0.0, 0.0]]))
        opposed = head(torch.tensor([[-0.2, 0.0, 0.0]]))
    assert aligned.item() > 0 > opposed.item()


def test_scale_is_learnable_and_positive() -> None:
    """Without a scale, cosine logits are bounded in [-1, 1] and cannot saturate."""

    head = CosineClassifierHead(8, initial_scale=30.0)
    assert head.log_scale.requires_grad
    assert head.log_scale.exp().item() == pytest.approx(30.0)
    # Parameterised in log space, so the scale can never become zero or negative.
    with torch.no_grad():
        head.log_scale.fill_(-20.0)
    assert head.log_scale.exp().item() > 0


# ---------------------------------------------------------------- parameter matching


def test_cosine_head_is_parameter_matched_to_the_linear_head() -> None:
    """Exactly one extra scalar, so capacity cannot explain any measured difference."""

    linear = sum(p.numel() for p in BinaryClassifierHead(768).parameters())
    cosine = sum(p.numel() for p in CosineClassifierHead(768).parameters())
    assert linear == 769
    assert cosine == 770
    assert cosine - linear == 1


def test_head_only_freezing_trains_the_whole_cosine_head_including_scale() -> None:
    backbone = CLIPVisionBackbone("mock", encoder=MockEncoder())
    model = CLIPBinaryDetector(backbone, CosineClassifierHead(backbone.feature_dim))
    configure_trainable_layers(model, "head_only")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    # The learnable scale must be trained, otherwise the head is silently crippled.
    assert trainable == {
        "classifier.classifier.weight",
        "classifier.classifier.bias",
        "classifier.log_scale",
    }
    assert all("backbone" not in name for name in trainable)


# --------------------------------------------------------------------- opt-in selection


def test_builder_defaults_to_the_original_linear_head() -> None:
    """Existing configs and completed runs must be unaffected by this feature."""

    assert isinstance(build_classifier_head(16), BinaryClassifierHead)
    assert isinstance(build_classifier_head(16, head_type="linear"), BinaryClassifierHead)
    assert isinstance(build_classifier_head(16, head_type=HeadType.LINEAR), BinaryClassifierHead)


def test_builder_selects_the_cosine_head_when_asked() -> None:
    assert isinstance(build_classifier_head(16, head_type="cosine"), CosineClassifierHead)
    assert isinstance(build_classifier_head(16, head_type=HeadType.COSINE), CosineClassifierHead)


def test_builder_rejects_an_unknown_head_rather_than_falling_back() -> None:
    # A silent fallback to linear would let a mis-typed config report the original model
    # under the modified model's name.
    with pytest.raises(ValueError, match="unknown model.head_type"):
        build_classifier_head(16, head_type="mlp")


def test_cosine_head_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="input_dim must be a positive integer"):
        CosineClassifierHead(0)
    with pytest.raises(ValueError, match="dropout must be in"):
        CosineClassifierHead(8, dropout=1.0)
    with pytest.raises(ValueError, match="initial_scale must be positive"):
        CosineClassifierHead(8, initial_scale=0.0)
    with pytest.raises(ValueError, match="embeddings must have shape"):
        CosineClassifierHead(8)(torch.randn(2, 5))


# -------------------------------------------------------------------- config validation


def _config(**model_overrides: object) -> dict[str, object]:
    """A minimal config that passes validation, for exercising model.head_type only."""

    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["model"].update(model_overrides)
    base["generators"] = {
        "train": ["adm"],
        "validation": ["adm"],
        "test": ["adm"],
        "include_real_images": True,
    }
    base["experiment"] = {"type": "baseline", "name": "unit_test"}
    return base


def test_config_accepts_both_head_types_and_rejects_others() -> None:
    validate_config(_config(head_type="linear"))
    validate_config(_config(head_type="cosine"))
    with pytest.raises(ValueError, match="model.head_type must be linear or cosine"):
        validate_config(_config(head_type="mlp"))


def test_config_without_head_type_is_still_valid() -> None:
    """Configs written before this feature existed must keep working unchanged."""

    config = _config()
    del config["model"]["head_type"]
    validate_config(config)
