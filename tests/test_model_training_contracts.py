"""Offline detector shape, freezing, training, validation, and checkpoint tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.models.checkpointing import (
    CheckpointMetadata,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.models.clip_detector import (
    BinaryClassifierHead,
    CLIPBinaryDetector,
    CLIPVisionBackbone,
    configure_trainable_layers,
)
from src.training.engine import train_one_epoch, validate_one_epoch


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


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(-1)


def detector() -> CLIPBinaryDetector:
    backbone = CLIPVisionBackbone("mock", encoder=MockEncoder())
    return CLIPBinaryDetector(backbone, BinaryClassifierHead(backbone.feature_dim))


def test_detector_returns_one_logit_per_image_including_batch_size_one() -> None:
    model = detector()
    assert model(torch.randn(4, 3, 8, 8)).logits.shape == (4,)
    assert model(torch.randn(1, 3, 8, 8)).logits.shape == (1,)
    probabilities = model.predict_proba(torch.randn(2, 3, 8, 8))
    assert probabilities.shape == (2,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_each_fine_tune_mode_exposes_exact_intended_parameters() -> None:
    model = detector()
    configure_trainable_layers(model, "head_only")
    head = {name for name, value in model.named_parameters() if value.requires_grad}
    assert head and all(name.startswith("classifier.") for name in head)
    configure_trainable_layers(model, "last_block")
    last = {name for name, value in model.named_parameters() if value.requires_grad}
    assert head < last
    assert any("layers.1" in name for name in last)
    configure_trainable_layers(model, "full")
    full = {name for name, value in model.named_parameters() if value.requires_grad}
    assert last < full
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    assert all(
        parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"]
    )


def batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "pixel_values": torch.tensor([[1.0], [-1.0], [0.8], [-0.8]]),
            "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
    ]


def test_validation_changes_no_model_or_optimizer_state() -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before_model = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    before_optimizer = optimizer.state_dict()
    result = validate_one_epoch(
        model=model, data_loader=batches(), loss_fn=nn.BCEWithLogitsLoss(), device="cpu"
    )
    assert result.sample_count == 4
    assert model.training
    assert all(
        torch.equal(before_model[name], tensor) for name, tensor in model.state_dict().items()
    )
    assert before_optimizer == optimizer.state_dict()


def test_checkpoint_round_trip_preserves_fixed_batch_predictions(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    fixed = torch.tensor([[0.2], [0.8]])
    expected = model(fixed).detach()
    checkpoint = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        metadata=CheckpointMetadata(1, "f1", 0.5, "tiny", "full", 42),
        resolved_config={"test": True},
    )
    path = tmp_path / "model.pt"
    save_checkpoint(checkpoint, path)
    restored = nn.Linear(1, 1)
    restored.load_state_dict(load_checkpoint(path)["model_state"])
    torch.testing.assert_close(restored(fixed), expected)
    assert path.with_suffix(".pt.sha256").is_file()


def test_tiny_batch_can_be_overfit() -> None:
    torch.manual_seed(3)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
    loss_fn = nn.BCEWithLogitsLoss()
    first = validate_one_epoch(
        model=model, data_loader=batches(), loss_fn=loss_fn, device="cpu"
    ).loss
    for _ in range(40):
        train_one_epoch(
            model=model, data_loader=batches(), optimizer=optimizer, loss_fn=loss_fn, device="cpu"
        )
    final = validate_one_epoch(
        model=model, data_loader=batches(), loss_fn=loss_fn, device="cpu"
    ).loss
    assert final < first * 0.25
