"""Checkpoint save and load (§26).

A checkpoint is a *directory*, and it is self-contained: weights, the exact model
config, the tokenizer, and the training state needed to resume. That is what lets
the inference server load one path and serve, with no network and no other files
(§2).

Loading uses `weights_only=True`, so a checkpoint file cannot execute code when it
is opened (§53). Everything stored is therefore restricted to tensors and plain
data — the NumPy RNG state is converted to lists on the way in for that reason.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bravien.model.config import BravienConfig
from bravien.model.model import BravienForCausalLM
from bravien.utils.logging import get_logger
from bravien.utils.paths import ensure_dir, safe_join

logger = get_logger("training.checkpoint")

#: Bumped when the on-disk layout changes incompatibly.
CHECKPOINT_FORMAT_VERSION = 1

WEIGHTS_NAME = "model.pt"
CONFIG_NAME = "config.json"
TRAINING_STATE_NAME = "training_state.pt"
METADATA_NAME = "metadata.json"
TOKENIZER_DIR = "tokenizer"
LATEST_POINTER = "latest.json"


class CheckpointError(RuntimeError):
    pass


@dataclass
class CheckpointMetadata:
    """Everything needed to say what this checkpoint *is* (§28)."""

    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    stage: str = "pretrain"
    run_name: str = "bravien"
    created_at: str = ""
    format_version: int = CHECKPOINT_FORMAT_VERSION
    model_name: str = ""
    parameters: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def _serialisable_rng_state() -> dict[str, Any]:
    """Capture RNG streams as tensors and plain lists.

    NumPy's state contains an ndarray, which `torch.load(weights_only=True)`
    refuses; converting to a list keeps the checkpoint loadable under that
    restriction.
    """
    numpy_state = np.random.get_state(legacy=False)
    bit_state = numpy_state.get("state", {})
    state: dict[str, Any] = {
        "python": json.dumps(random.getstate()[1]),
        "python_pos": random.getstate()[2],
        "numpy_bit_generator": numpy_state.get("bit_generator", "MT19937"),
        "numpy_key": [int(x) for x in np.asarray(bit_state.get("key", []))],
        "numpy_pos": int(bit_state.get("pos", 0)),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore what `_serialisable_rng_state` captured, tolerating gaps."""
    try:
        python_state = tuple(json.loads(state["python"]))
        random.setstate((3, python_state, state["python_pos"]))
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("could not restore Python RNG state: %s", exc)

    try:
        key = np.asarray(state["numpy_key"], dtype=np.uint32)
        np.random.set_state(
            {
                "bit_generator": state.get("numpy_bit_generator", "MT19937"),
                "state": {"key": key, "pos": int(state["numpy_pos"])},
            }
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("could not restore NumPy RNG state: %s", exc)

    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))

    cuda_states = state.get("cuda")
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cuda_states])
        else:
            logger.warning(
                "checkpoint holds %d CUDA RNG state(s) but %d device(s) are "
                "present; CUDA streams not restored",
                len(cuda_states),
                torch.cuda.device_count(),
            )


#: Backoff between attempts at a directory rename or removal, in seconds.
#:
#: Renaming a directory on Windows fails with `ERROR_ACCESS_DENIED` while any
#: process holds a handle to a file inside it. A checkpoint is tens of megabytes
#: and this repository's usual home is a synced folder, so the sync client and the
#: virus scanner behind it both open `model.pt` in the moment between writing the
#: file and renaming its directory — and the save loses a race it wins a second
#: later. Retrying is the fix: the bytes on disk were already complete.
_RETRY_DELAYS = (0.1, 0.3, 0.7, 1.5, 3.0)


def _rename_with_retry(source: Path, target: Path) -> None:
    """Rename `source` to `target`, waiting out a transient lock."""
    for delay in _RETRY_DELAYS:
        try:
            source.replace(target)
            return
        except OSError as exc:
            logger.debug(
                "rename %s -> %s blocked (%s); retrying in %.1fs",
                source.name,
                target.name,
                exc,
                delay,
            )
            time.sleep(delay)

    try:
        source.replace(target)
    except OSError as exc:
        raise CheckpointError(
            f"could not rename {source} to {target} after "
            f"{len(_RETRY_DELAYS) + 1} attempts: {exc}\n"
            f"Something is holding the directory open — a file browser, a sync "
            f"client, or an antivirus scan. The written data is intact in {source}."
        ) from exc


def _remove_directory(path: Path) -> None:
    """Delete a directory tree, waiting out a transient lock.

    Deliberately not `rmtree(ignore_errors=True)`: silently leaving a
    half-deleted directory behind produces something still named `step-*` with no
    weights in it, which looks like a checkpoint to anything scanning the run
    directory. Failing loudly is more useful than that.
    """
    if not path.exists():
        return

    for delay in _RETRY_DELAYS:
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            logger.debug(
                "removal of %s blocked (%s); retrying in %.1fs", path.name, exc, delay
            )
            time.sleep(delay)

    shutil.rmtree(path)


def _promote_directory(staging: Path, target: Path) -> None:
    """Move a finished staging directory into its final name.

    Any directory already at `target` is moved aside rather than deleted first, so
    a rename that fails anyway leaves the previous checkpoint recoverable instead
    of destroying it on the way to writing nothing.
    """
    displaced: Path | None = None
    if target.exists():
        displaced = target.with_name(target.name + ".old")
        _remove_directory(displaced)
        _rename_with_retry(target, displaced)

    try:
        _rename_with_retry(staging, target)
    except BaseException:
        if displaced is not None:
            try:
                _rename_with_retry(displaced, target)
            except OSError as exc:
                logger.error(
                    "could not restore the previous checkpoint; it is at %s (%s)",
                    displaced,
                    exc,
                )
        raise

    if displaced is not None:
        _remove_directory(displaced)


def save_checkpoint(
    directory: str | Path,
    model: BravienForCausalLM,
    *,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    tokenizer: Any | None = None,
    save_rng: bool = True,
) -> Path:
    """Write a complete checkpoint directory.

    Written to a sibling `.tmp` directory and renamed, so an interrupted save
    cannot leave a half-written checkpoint that looks valid.
    """
    from bravien.data.manifest import utc_now

    directory = Path(directory)
    staging = directory.with_name(directory.name + ".tmp")
    _remove_directory(staging)
    ensure_dir(staging)

    metadata.created_at = metadata.created_at or utc_now()
    metadata.model_name = model.config.name
    metadata.parameters = model.parameter_report().total
    metadata.format_version = CHECKPOINT_FORMAT_VERSION
    metadata.environment.setdefault("torch", torch.__version__)
    metadata.environment.setdefault("cuda", torch.version.cuda)

    # Weights on CPU: a checkpoint saved from cuda:0 must load on a machine with
    # no GPU at all.
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, staging / WEIGHTS_NAME)
    model.config.save(staging / CONFIG_NAME)

    if optimizer is not None or scheduler is not None or save_rng:
        training_state: dict[str, Any] = {}
        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            training_state["scheduler"] = scheduler.state_dict()
        if save_rng:
            training_state["rng"] = _serialisable_rng_state()
        training_state["step"] = metadata.step
        training_state["tokens_seen"] = metadata.tokens_seen
        torch.save(training_state, staging / TRAINING_STATE_NAME)

    if tokenizer is not None:
        tokenizer.save_pretrained(staging / TOKENIZER_DIR)
        metadata.tokenizer = {
            "vocab_size": tokenizer.vocab_size,
            "checksum": tokenizer.metadata.get("vocab_checksum"),
            "bundled": True,
        }
        if tokenizer.vocab_size != model.config.vocab_size:
            raise CheckpointError(
                f"tokenizer vocab ({tokenizer.vocab_size}) does not match model "
                f"vocab ({model.config.vocab_size}); the checkpoint would "
                f"generate ids the tokenizer cannot decode"
            )

    (staging / METADATA_NAME).write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _promote_directory(staging, directory)
    logger.info("saved checkpoint %s (step %d)", directory, metadata.step)
    return directory


@dataclass
class LoadedCheckpoint:
    model: BravienForCausalLM
    config: BravienConfig
    metadata: CheckpointMetadata
    tokenizer: Any | None = None
    training_state: dict[str, Any] | None = None
    path: Path = Path()


def load_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    load_tokenizer: bool = True,
    load_training_state: bool = False,
    strict: bool = True,
) -> LoadedCheckpoint:
    """Load a checkpoint directory.

    Args:
        load_training_state: also read optimizer/scheduler/RNG. Inference does
            not need it, and skipping it avoids reading a file several times the
            size of the weights.
        strict: require the state dict to match the model exactly.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CheckpointError(f"not a checkpoint directory: {directory}")

    config_path = directory / CONFIG_NAME
    weights_path = directory / WEIGHTS_NAME
    for path in (config_path, weights_path):
        if not path.exists():
            raise CheckpointError(f"checkpoint is missing {path.name}: {directory}")

    metadata_path = directory / METADATA_NAME
    metadata = (
        CheckpointMetadata.from_dict(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
        if metadata_path.exists()
        else CheckpointMetadata()
    )
    if metadata.format_version > CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint format v{metadata.format_version} is newer than this "
            f"build supports (v{CHECKPOINT_FORMAT_VERSION})"
        )

    config = BravienConfig.load(config_path)
    model = BravienForCausalLM(config)

    # weights_only=True: refuses to unpickle anything but tensors and plain data,
    # so opening an untrusted checkpoint cannot run code.
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if config.tie_word_embeddings:
        # The tied head is not an independent parameter. Older or foreign state
        # dicts may or may not include it; dropping it keeps both loadable.
        state_dict.pop("lm_head.weight", None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    if strict and (missing or unexpected):
        raise CheckpointError(
            f"state dict does not match the model.\n"
            f"  missing:    {missing}\n"
            f"  unexpected: {unexpected}"
        )
    if missing or unexpected:
        logger.warning(
            "loaded with missing=%s unexpected=%s", missing, unexpected
        )

    model.to(device)

    tokenizer = None
    if load_tokenizer and (directory / TOKENIZER_DIR).is_dir():
        from bravien.tokenizer.tokenizer import BravienTokenizer

        tokenizer = BravienTokenizer.from_pretrained(directory / TOKENIZER_DIR)
        if tokenizer.vocab_size != config.vocab_size:
            raise CheckpointError(
                f"bundled tokenizer vocab ({tokenizer.vocab_size}) does not "
                f"match the model ({config.vocab_size})"
            )

    training_state = None
    if load_training_state:
        state_path = directory / TRAINING_STATE_NAME
        if state_path.exists():
            training_state = torch.load(
                state_path, map_location="cpu", weights_only=True
            )
        else:
            logger.warning(
                "%s has no %s; training will restart from step 0",
                directory.name,
                TRAINING_STATE_NAME,
            )

    return LoadedCheckpoint(
        model=model,
        config=config,
        metadata=metadata,
        tokenizer=tokenizer,
        training_state=training_state,
        path=directory,
    )


def restore_training_state(
    state: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    restore_rng: bool = True,
) -> int:
    """Apply a loaded training state. Returns the step to resume from."""
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state["scheduler"])
    if restore_rng and "rng" in state:
        _restore_rng_state(state["rng"])
    return int(state.get("step", 0))


# ------------------------------------------------------------------ management


def write_latest_pointer(run_dir: str | Path, checkpoint_dir: str | Path) -> Path:
    """Record which checkpoint is newest.

    A file rather than a symlink: Windows needs elevated rights to create
    symlinks, and this has to work there (§57).
    """
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    pointer = run_dir / LATEST_POINTER
    pointer.write_text(
        json.dumps({"latest": Path(checkpoint_dir).name}, indent=2), encoding="utf-8"
    )
    return pointer


def resolve_latest(run_dir: str | Path) -> Path | None:
    """Find the newest checkpoint in a run directory.

    Prefers the pointer file, falls back to the highest `step-*` directory so a
    run whose pointer was lost is still resumable.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None

    pointer = run_dir / LATEST_POINTER
    if pointer.exists():
        try:
            name = json.loads(pointer.read_text(encoding="utf-8"))["latest"]
            # safe_join: the pointer is a file on disk and could name "..".
            candidate = safe_join(run_dir, name)
            if candidate.is_dir():
                return candidate
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("ignoring unreadable %s: %s", pointer, exc)

    steps = [
        d
        for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("step-") and not d.name.endswith(".tmp")
    ]
    if not steps:
        return None

    def step_of(path: Path) -> int:
        try:
            return int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(steps, key=step_of)


def checkpoint_dir_for(run_dir: str | Path, step: int) -> Path:
    """Zero-padded so directories sort correctly in a file listing."""
    return Path(run_dir) / f"step-{step:07d}"


def prune_checkpoints(run_dir: str | Path, keep: int, *, protect: set[str] | None = None) -> list[Path]:
    """Delete all but the newest `keep` checkpoints. Returns what was removed.

    Checkpoints are large and this repository may sit inside a synced folder, so
    unbounded retention is a real problem (§67).
    """
    if keep <= 0:
        return []
    run_dir = Path(run_dir)
    protect = protect or set()

    candidates = sorted(
        (
            d
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("step-") and d.name not in protect
        ),
        key=lambda d: d.name,
    )
    removed: list[Path] = []
    for path in candidates[:-keep] if len(candidates) > keep else []:
        try:
            _remove_directory(path)
        except OSError as exc:
            # A checkpoint that outlives its welcome costs disk space. That is not
            # a reason to end a training run that is otherwise going fine.
            logger.warning("could not prune %s: %s", path.name, exc)
            continue
        removed.append(path)
        logger.info("pruned old checkpoint %s", path.name)
    return removed
