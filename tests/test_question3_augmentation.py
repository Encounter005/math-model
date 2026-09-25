import torch

from scripts.question3.augmentation import make_augmented_view


def _batch(text_mask: torch.Tensor, audio_mask: torch.Tensor, vision_mask: torch.Tensor):
    return {
        "id": ["01"],
        "raw_text": ["sample"],
        "input_ids": torch.ones(1, 50, dtype=torch.long),
        "attention_mask": text_mask.long(),
        "audio": torch.ones(1, 50, 74),
        "vision": torch.ones(1, 50, 35),
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "vision_mask": vision_mask,
        "class_label": None,
        "regression_label": None,
    }


def _has_only_contiguous_hidden_runs(
    natural_mask: torch.Tensor, augmented_mask: torch.Tensor
) -> bool:
    hidden_positions = torch.where(natural_mask & ~augmented_mask)[1]
    return bool(
        hidden_positions.numel()
        and torch.equal(
            hidden_positions,
            torch.arange(hidden_positions[0], hidden_positions[-1] + 1),
        )
    )


def test_augmented_mask_hides_a_contiguous_valid_span_reproducibly():
    text_mask = torch.ones(1, 50, dtype=torch.bool)
    batch = _batch(text_mask, torch.zeros_like(text_mask), torch.zeros_like(text_mask))

    first = make_augmented_view(batch, seed=42, rates=(0.2,), feature_dropout=0)
    second = make_augmented_view(batch, seed=42, rates=(0.2,), feature_dropout=0)

    assert torch.equal(first["text_mask"], second["text_mask"])
    assert _has_only_contiguous_hidden_runs(batch["text_mask"], first["text_mask"])


def test_augmentation_never_marks_padding_as_observed():
    text_mask = torch.tensor([[True] * 10 + [False] * 40])
    batch_with_padding = _batch(text_mask, text_mask.clone(), text_mask.clone())

    augmented = make_augmented_view(
        batch_with_padding, seed=42, rates=(0.3,), feature_dropout=0.05
    )

    assert not augmented["audio_mask"][~batch_with_padding["text_mask"]].any()


def test_augmentation_does_not_mutate_the_source_batch():
    text_mask = torch.ones(1, 50, dtype=torch.bool)
    batch = _batch(text_mask, text_mask.clone(), text_mask.clone())
    original_audio = batch["audio"].clone()
    original_text_mask = batch["text_mask"].clone()

    make_augmented_view(batch, seed=42, rates=(0.2,), feature_dropout=1.0)

    assert torch.equal(batch["audio"], original_audio)
    assert torch.equal(batch["text_mask"], original_text_mask)
