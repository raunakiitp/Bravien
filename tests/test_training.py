"""Training tests: schedule, parameter groups, loss masking, checkpoints (§64).

Two of these are load-bearing beyond their size.

`test_loss_is_invariant_to_right_padding` proves the reason `Trainer.train_step`
does not pass an attention mask: under right padding and causal masking, a real
token never attends to a pad, and `ignore_index` keeps pads out of the loss mean.
That is an assumption the training loop relies on rather than checks, so it is
checked here.

`test_promote_survives_a_locked_directory` covers a save that used to fail
outright on Windows, and `test_a_failed_promotion_leaves_the_previous_checkpoint`
covers the thing that would be much worse than failing: losing good weights while
trying to write new ones.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import torch

from bravien.model.model import IGNORE_INDEX, BravienForCausalLM
from bravien.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointMetadata,
    _promote_directory,
    _rename_with_retry,
    checkpoint_dir_for,
    load_checkpoint,
    prune_checkpoints,
    resolve_latest,
    restore_training_state,
    save_checkpoint,
    write_latest_pointer,
)
from bravien.training.optimizer import build_optimizer, split_parameters
from bravien.training.scheduler import SchedulerConfig, lr_at_step
from bravien.training.trainer import TrainingConfig

# -------------------------------------------------------------------- schedule


def test_warmup_ramps_from_near_zero_to_the_base_rate():
    """Warmup counts steps as 1-based, so step 0 trains rather than stalling."""
    config = SchedulerConfig(warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
    assert lr_at_step(0, 1e-3, config) == pytest.approx(1e-5)
    assert lr_at_step(49, 1e-3, config) == pytest.approx(5e-4)
    # The last warmup step is the 100th, i.e. index 99.
    assert lr_at_step(99, 1e-3, config) == pytest.approx(1e-3)


def test_the_rate_decays_to_the_floor_and_stays_there():
    config = SchedulerConfig(warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert lr_at_step(100, 1e-3, config) == pytest.approx(1e-4)
    # Past the end of the schedule the floor holds rather than going negative.
    assert lr_at_step(500, 1e-3, config) == pytest.approx(1e-4)


def test_the_rate_never_leaves_the_configured_band():
    config = SchedulerConfig(warmup_steps=37, total_steps=400, min_lr_ratio=0.05)
    rates = [lr_at_step(step, 2e-4, config) for step in range(0, 500)]
    assert min(rates) >= 0.0
    assert max(rates) == pytest.approx(2e-4)
    assert all(rate <= 2e-4 + 1e-12 for rate in rates)


def test_decay_is_monotonic_after_warmup():
    config = SchedulerConfig(warmup_steps=20, total_steps=200, min_lr_ratio=0.1)
    after = [lr_at_step(s, 1e-3, config) for s in range(20, 201)]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:]))


# ------------------------------------------------------------ parameter groups


def test_norms_and_biases_are_excluded_from_weight_decay(tiny_model):
    """Decaying a gain or a bias shrinks it toward zero for no good reason."""
    decay, no_decay = split_parameters(tiny_model)
    assert decay and no_decay

    decayed = {id(p) for p in decay}
    for name, param in tiny_model.named_parameters():
        is_matrix = param.dim() >= 2
        assert (id(param) in decayed) == is_matrix, name


def test_every_parameter_lands_in_exactly_one_group(tiny_model):
    decay, no_decay = split_parameters(tiny_model)
    total = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
    trainable = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    assert total == trainable


def test_the_optimizer_receives_both_groups(tiny_model):
    from bravien.training.optimizer import OptimizerConfig

    optimizer = build_optimizer(tiny_model, OptimizerConfig(weight_decay=0.1))
    decays = {group["weight_decay"] for group in optimizer.param_groups}
    assert decays == {0.1, 0.0}


# ------------------------------------------------------------------- precision


@pytest.mark.parametrize("alias", ["auto", "fp32", "fp16", "bf16"])
def test_the_trainer_accepts_its_own_precision_spelling(alias):
    assert TrainingConfig(precision=alias).precision == alias


@pytest.mark.parametrize("spelling", ["float32", "float16", "bfloat16", "half", ""])
def test_torch_dtype_spellings_are_rejected(spelling):
    """The scripts translate; the config does not guess.

    `scripts/pretrain.py` once passed these through unmapped, so every explicit
    `--precision` was rejected and only the default worked.
    """
    with pytest.raises(ValueError):
        TrainingConfig(precision=spelling)


def test_the_scripts_map_every_choice_they_offer():
    """Whatever the CLI advertises must be a spelling the config accepts."""
    import scripts.finetune as finetune
    import scripts.pretrain as pretrain

    for module in (pretrain, finetune):
        for exposed, internal in module.PRECISION_NAMES.items():
            assert TrainingConfig(precision=internal).precision == internal, exposed


# ------------------------------------------------------------- the loss masking


def _padded_batch(model, pad_id=0):
    """One short and one long sequence, right-padded, as `collate_sft` builds."""
    long_ids = [5, 6, 7, 8, 9, 10]
    short_ids = [11, 12, 13]
    pad = [pad_id] * (len(long_ids) - len(short_ids))

    input_ids = torch.tensor([long_ids, short_ids + pad])
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 6, 7, 8, 9, 10],
            [IGNORE_INDEX, 12, 13] + [IGNORE_INDEX] * len(pad),
        ]
    )
    return input_ids, labels


def test_loss_is_invariant_to_right_padding(tiny_model):
    """Why `train_step` needs no attention mask.

    Causal masking means a real token at position i attends only to j <= i, all of
    which are real under right padding. The pad positions do produce logits, but
    their labels are `IGNORE_INDEX`, and cross-entropy's mean reduction averages
    only over non-ignored elements. So padding cannot move the loss — and omitting
    the mask lets the fused causal kernel run.
    """
    input_ids, labels = _padded_batch(tiny_model)

    with torch.no_grad():
        batched = tiny_model(input_ids, labels=labels).loss

        # The same two sequences scored one at a time, with no padding at all.
        losses, counts = [], []
        for row in range(2):
            keep = labels[row] != IGNORE_INDEX
            length = int(keep.nonzero().max()) + 1
            out = tiny_model(
                input_ids[row : row + 1, :length],
                labels=labels[row : row + 1, :length],
            )
            supervised = int(keep[:length].sum())
            losses.append(out.loss * supervised)
            counts.append(supervised)

        unpadded = sum(losses) / sum(counts)

    assert batched.item() == pytest.approx(unpadded.item(), rel=1e-5)


def test_padding_amount_does_not_change_the_loss(tiny_model):
    """The same batch padded to 8 and to 16 must score identically."""
    input_ids, labels = _padded_batch(tiny_model)
    extra = 6
    wide_ids = torch.cat([input_ids, torch.zeros(2, extra, dtype=torch.long)], dim=1)
    wide_labels = torch.cat(
        [labels, torch.full((2, extra), IGNORE_INDEX, dtype=torch.long)], dim=1
    )

    with torch.no_grad():
        narrow = tiny_model(input_ids, labels=labels).loss
        wide = tiny_model(wide_ids, labels=wide_labels).loss

    assert narrow.item() == pytest.approx(wide.item(), rel=1e-5)


def test_an_all_masked_batch_is_what_ignore_index_cannot_save_you_from(tiny_model):
    """Documents why `build_sft_example` drops unsupervised examples upstream."""
    input_ids = torch.tensor([[5, 6, 7]])
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    with torch.no_grad():
        loss = tiny_model(input_ids, labels=labels).loss
    assert torch.isnan(loss)


# ------------------------------------------------------------------ checkpoints


@pytest.fixture
def saved(tmp_path, tiny_config, trained_tokenizer):
    """A checkpoint on disk, with a tokenizer whose vocab matches the model."""
    config = tiny_config.replace(vocab_size=trained_tokenizer.vocab_size)
    torch.manual_seed(0)
    model = BravienForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # One step, so the optimizer state is non-empty and worth round-tripping.
    model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])).loss.backward()
    optimizer.step()

    path = save_checkpoint(
        tmp_path / "run" / "step-0000010",
        model,
        metadata=CheckpointMetadata(step=10, stage="sft", tokens_seen=1234),
        optimizer=optimizer,
        tokenizer=trained_tokenizer,
    )
    return path, model


def test_a_checkpoint_is_self_contained(saved):
    """One directory, no network, no other files — that is what §2 requires."""
    path, _ = saved
    for name in ("model.pt", "config.json", "metadata.json", "training_state.pt"):
        assert (path / name).is_file(), name
    assert (path / "tokenizer" / "tokenizer.json").is_file()


def test_loading_restores_the_same_weights(saved):
    path, original = saved
    loaded = load_checkpoint(path)
    for (name, before), (_, after) in zip(
        original.state_dict().items(), loaded.model.state_dict().items()
    ):
        assert torch.equal(before.cpu(), after.cpu()), name


def test_metadata_survives_the_round_trip(saved):
    path, _ = saved
    metadata = load_checkpoint(path).metadata
    assert metadata.step == 10
    assert metadata.stage == "sft"
    assert metadata.tokens_seen == 1234
    assert metadata.format_version == CHECKPOINT_FORMAT_VERSION
    assert metadata.parameters > 0
    assert metadata.created_at


def test_weights_are_stored_on_the_cpu(saved):
    """A checkpoint written from a GPU has to load on a machine without one."""
    path, _ = saved
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in state.values())


def test_the_training_state_resumes_the_step_counter(saved):
    path, _ = saved
    loaded = load_checkpoint(path, load_training_state=True)
    assert loaded.training_state is not None
    assert restore_training_state(loaded.training_state) == 10


def test_inference_can_skip_the_training_state(saved):
    """Serving does not need the optimizer, which is larger than the weights."""
    path, _ = saved
    assert load_checkpoint(path, load_training_state=False).training_state is None


def test_a_mismatched_tokenizer_is_refused(tmp_path, tiny_config, trained_tokenizer):
    """Otherwise the model emits ids the tokenizer cannot decode."""
    model = BravienForCausalLM(tiny_config.replace(vocab_size=64))
    with pytest.raises(CheckpointError, match="does not match"):
        save_checkpoint(
            tmp_path / "bad",
            model,
            metadata=CheckpointMetadata(),
            tokenizer=trained_tokenizer,
        )


def test_a_future_format_version_is_refused(saved):
    path, _ = saved
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    metadata["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CheckpointError, match="newer than this build"):
        load_checkpoint(path)


def test_a_directory_that_is_not_a_checkpoint_is_refused(tmp_path):
    with pytest.raises(CheckpointError, match="not a checkpoint directory"):
        load_checkpoint(tmp_path / "nothing-here")


def test_a_checkpoint_missing_its_weights_is_refused(saved):
    path, _ = saved
    (path / "model.pt").unlink()
    with pytest.raises(CheckpointError, match="missing model.pt"):
        load_checkpoint(path)


def test_no_staging_directory_is_left_behind(saved):
    path, _ = saved
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob("*.old"))


# ------------------------------------------------- atomic promotion on Windows


def test_promote_into_a_free_name(tmp_path):
    staging = tmp_path / "step-1.tmp"
    staging.mkdir()
    (staging / "model.pt").write_text("weights")

    _promote_directory(staging, tmp_path / "step-1")

    assert (tmp_path / "step-1" / "model.pt").read_text() == "weights"
    assert not staging.exists()


def test_promote_over_an_existing_checkpoint(tmp_path):
    """Reachable by re-running a fresh job that lands on the same step number."""
    target = tmp_path / "step-1"
    target.mkdir()
    (target / "model.pt").write_text("old")

    staging = tmp_path / "step-1.tmp"
    staging.mkdir()
    (staging / "model.pt").write_text("new")

    _promote_directory(staging, target)

    assert (target / "model.pt").read_text() == "new"
    assert not (tmp_path / "step-1.old").exists()


def test_promote_survives_a_locked_directory(tmp_path):
    """The failure this retry exists for.

    A directory rename on Windows is denied while any process holds a handle
    inside it, and a sync client or virus scanner opens a freshly written weights
    file immediately. Renaming a moment later succeeds.
    """
    staging = tmp_path / "step-2.tmp"
    staging.mkdir()
    weights = staging / "model.pt"
    weights.write_bytes(b"x" * 2048)

    opened = threading.Event()

    def hold_it_briefly():
        with weights.open("rb"):
            opened.set()
            time.sleep(0.5)

    holder = threading.Thread(target=hold_it_briefly)
    holder.start()
    opened.wait(timeout=5)
    try:
        _promote_directory(staging, tmp_path / "step-2")
    finally:
        holder.join()

    assert (tmp_path / "step-2" / "model.pt").stat().st_size == 2048


def test_a_failed_promotion_leaves_the_previous_checkpoint(tmp_path, monkeypatch):
    """Worse than a failed save: a failed save that also destroyed good weights."""
    target = tmp_path / "step-3"
    target.mkdir()
    (target / "model.pt").write_text("precious")

    staging = tmp_path / "step-3.tmp"
    staging.mkdir()
    (staging / "model.pt").write_text("doomed")

    real = _rename_with_retry

    def fail_only_the_promotion(source, destination):
        if source.name.endswith(".tmp"):
            raise OSError(5, "simulated lock")
        return real(source, destination)

    monkeypatch.setattr(
        "bravien.training.checkpoint._rename_with_retry", fail_only_the_promotion
    )
    with pytest.raises(OSError):
        _promote_directory(staging, target)

    assert (target / "model.pt").read_text() == "precious"


def test_rename_gives_up_with_an_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setattr("bravien.training.checkpoint._RETRY_DELAYS", (0.0, 0.0))

    staging = tmp_path / "step-4.tmp"
    staging.mkdir()

    def always_denied(self, target):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr("pathlib.Path.replace", always_denied)
    with pytest.raises(CheckpointError, match="holding the directory open"):
        _rename_with_retry(staging, tmp_path / "step-4")


# -------------------------------------------------------------- run management


def test_the_pointer_names_the_newest_checkpoint(tmp_path):
    """A file rather than a symlink: Windows needs privileges for symlinks."""
    for step in (100, 200):
        checkpoint_dir_for(tmp_path, step).mkdir(parents=True)
    write_latest_pointer(tmp_path, checkpoint_dir_for(tmp_path, 200))

    assert resolve_latest(tmp_path) == checkpoint_dir_for(tmp_path, 200)


def test_the_highest_step_wins_when_the_pointer_is_gone(tmp_path):
    for step in (100, 900, 200):
        checkpoint_dir_for(tmp_path, step).mkdir(parents=True)
    assert resolve_latest(tmp_path) == checkpoint_dir_for(tmp_path, 900)


def test_an_unreadable_pointer_falls_back_instead_of_failing(tmp_path):
    checkpoint_dir_for(tmp_path, 300).mkdir(parents=True)
    (tmp_path / "latest.json").write_text("{ not json", encoding="utf-8")
    assert resolve_latest(tmp_path) == checkpoint_dir_for(tmp_path, 300)


def test_a_pointer_cannot_escape_the_run_directory(tmp_path):
    """The pointer is a file on disk, so it could name `..`."""
    checkpoint_dir_for(tmp_path, 300).mkdir(parents=True)
    (tmp_path / "latest.json").write_text(
        json.dumps({"latest": "../../elsewhere"}), encoding="utf-8"
    )
    assert resolve_latest(tmp_path) == checkpoint_dir_for(tmp_path, 300)


def test_a_half_written_checkpoint_is_never_selected(tmp_path):
    """`.tmp` directories are in-progress saves, not results."""
    checkpoint_dir_for(tmp_path, 100).mkdir(parents=True)
    (tmp_path / "step-0000200.tmp").mkdir()
    assert resolve_latest(tmp_path) == checkpoint_dir_for(tmp_path, 100)


def test_an_empty_run_directory_resolves_to_nothing(tmp_path):
    assert resolve_latest(tmp_path) is None
    assert resolve_latest(tmp_path / "never-existed") is None


def test_pruning_keeps_the_newest(tmp_path):
    for step in range(100, 600, 100):
        checkpoint_dir_for(tmp_path, step).mkdir(parents=True)

    removed = prune_checkpoints(tmp_path, keep=2)

    assert len(removed) == 3
    survivors = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert survivors == ["step-0000400", "step-0000500"]


def test_pruning_can_protect_a_checkpoint(tmp_path):
    """A fine-tune reads its base from disk; pruning must not delete it."""
    for step in range(100, 400, 100):
        checkpoint_dir_for(tmp_path, step).mkdir(parents=True)

    prune_checkpoints(tmp_path, keep=1, protect={"step-0000100"})

    assert checkpoint_dir_for(tmp_path, 100).is_dir()


def test_pruning_nothing_is_a_no_op(tmp_path):
    checkpoint_dir_for(tmp_path, 100).mkdir(parents=True)
    assert prune_checkpoints(tmp_path, keep=0) == []
    assert prune_checkpoints(tmp_path, keep=10) == []
    assert checkpoint_dir_for(tmp_path, 100).is_dir()


# ----------------------------------------------------------------- the SFT run


def test_an_sft_run_relabels_itself(tmp_path):
    """The stage reaches checkpoint metadata and the UI badge.

    Left at the `TrainingConfig` default it would label fine-tuned weights
    `pretrain`, which is the kind of quiet mislabelling §72 rules out.
    """
    from bravien.training.sft import SFTRun

    run = SFTRun(base_checkpoint=tmp_path, training=TrainingConfig())
    assert run.training.stage == "sft"


def test_an_explicit_stage_is_left_alone(tmp_path):
    from bravien.training.sft import SFTRun

    run = SFTRun(
        base_checkpoint=tmp_path, training=TrainingConfig(stage="preference")
    )
    assert run.training.stage == "preference"


def test_the_fine_tuning_rate_is_well_below_pretraining():
    """Inheriting pretraining's rate would wash the learned weights out."""
    from bravien.training.optimizer import OptimizerConfig
    from bravien.training.sft import DEFAULT_SFT_LR

    assert DEFAULT_SFT_LR < OptimizerConfig().learning_rate / 2


def test_resolving_a_base_accepts_a_run_directory(tmp_path, saved):
    """`--base checkpoints/bravien` should mean "the newest one in there"."""
    from bravien.training.sft import resolve_base_checkpoint

    path, _ = saved
    write_latest_pointer(path.parent, path)

    assert resolve_base_checkpoint(path) == path
    assert resolve_base_checkpoint(path.parent) == path


def test_resolving_an_empty_base_says_what_to_do(tmp_path):
    from bravien.training.sft import resolve_base_checkpoint

    (tmp_path / "empty-run").mkdir()
    with pytest.raises(CheckpointError):
        resolve_base_checkpoint(tmp_path / "empty-run")
