"""The training loop (§21–§24).

One loop serves pretraining and instruction tuning. The difference between them
lives entirely in the data — pretraining supervises every token, SFT masks the
prompt with `IGNORE_INDEX` — so there is no reason for two loops that can drift
apart.

What the loop guarantees:

* An optimizer step is taken after `grad_accum_steps` micro-batches, and the loss
  is scaled by `1/grad_accum_steps`, so changing the accumulation factor does not
  change the effective learning rate.
* Gradients are clipped by global norm *after* unscaling, which is the only point
  at which the norm is meaningful.
* A checkpoint written at step N can be resumed at step N with the optimizer,
  schedule and RNG streams intact.
* Every logged number is measured. Nothing here is estimated or filled in (§72).
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from bravien.data.dataset import batch_to_device
from bravien.model.model import IGNORE_INDEX, BravienForCausalLM
from bravien.training.checkpoint import (
    CheckpointMetadata,
    checkpoint_dir_for,
    prune_checkpoints,
    save_checkpoint,
    write_latest_pointer,
)
from bravien.training.optimizer import OptimizerConfig, build_optimizer
from bravien.training.scheduler import LearningRateSchedule, SchedulerConfig
from bravien.utils.hardware import DeviceInfo, detect_device
from bravien.utils.logging import Timer, format_count, format_duration, get_logger

logger = get_logger("training.trainer")


@dataclass
class TrainingConfig:
    """Everything the loop needs, in one serialisable object.

    Defaults are sized for a single 6 GB consumer GPU, because that is the machine
    this has to work on (§57).
    """

    run_name: str = "bravien"
    stage: str = "pretrain"
    output_dir: Path = Path("checkpoints")

    max_steps: int = 1000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    #: "auto" resolves to bf16 on hardware that supports it, fp16 on older CUDA,
    #: fp32 on CPU.
    precision: str = "auto"
    device: str | None = None

    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    save_every: int = 500
    keep_checkpoints: int = 3
    #: Also save on the final step even if it is not a multiple of save_every.
    save_at_end: bool = True

    seed: int = 0
    #: Abort if the loss becomes NaN or Inf rather than filling a checkpoint with
    #: dead weights.
    stop_on_nan: bool = True
    compile_model: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.scheduler.total_steps != self.max_steps:
            # A schedule that decays over a different horizon than the run is a
            # silent misconfiguration: the lr would either never reach the floor
            # or sit at it for most of training.
            self.scheduler = SchedulerConfig(
                warmup_steps=min(self.scheduler.warmup_steps, max(self.max_steps // 10, 1)),
                total_steps=self.max_steps,
                min_lr_ratio=self.scheduler.min_lr_ratio,
                kind=self.scheduler.kind,
            )

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


@dataclass
class StepMetrics:
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens: int
    supervised_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def perplexity(self) -> float:
        # Overflows to inf for an untrained model at high vocab; report it rather
        # than clamping, so the number stays honest.
        try:
            return math.exp(self.loss)
        except OverflowError:
            return float("inf")

    def format(self) -> str:
        return (
            f"step {self.step:>6}  loss {self.loss:8.4f}  ppl {self.perplexity:>10.2f}  "
            f"lr {self.learning_rate:.3e}  gnorm {self.grad_norm:7.3f}  "
            f"{format_count(int(self.tokens_per_second)):>8} tok/s"
        )


def resolve_precision(requested: str, device: DeviceInfo) -> tuple[torch.dtype, bool]:
    """Pick the autocast dtype and whether a GradScaler is needed.

    bf16 needs no loss scaling: it has fp32's exponent range, so gradients do not
    underflow to zero. fp16 does need it. Returning both together keeps that
    coupling in one place.
    """
    if requested == "fp32" or not device.is_cuda:
        return torch.float32, False
    if requested == "bf16":
        return torch.bfloat16, False
    if requested == "fp16":
        return torch.float16, True
    # auto
    if device.supports_bf16:
        return torch.bfloat16, False
    return torch.float16, True


def _cycle(loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever, restarting the loader at each epoch boundary."""
    epoch = 0
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        epoch += 1
        if not yielded:
            raise RuntimeError("training dataloader yielded no batches")
        logger.debug("dataloader epoch %d complete", epoch)


class Trainer:
    """Owns the loop, the optimizer, and checkpointing."""

    def __init__(
        self,
        model: BravienForCausalLM,
        config: TrainingConfig,
        train_loader: DataLoader,
        *,
        eval_loader: DataLoader | None = None,
        tokenizer: Any | None = None,
        dataset_info: dict[str, Any] | None = None,
        device: DeviceInfo | None = None,
    ) -> None:
        self.config = config
        self.device_info = device or detect_device(config.device)
        self.device = torch.device(self.device_info.device)
        self.dtype, self.needs_scaler = resolve_precision(
            config.precision, self.device_info
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.dataset_info = dataset_info or {}

        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.schedule = LearningRateSchedule(self.optimizer, config.scheduler)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.needs_scaler and self.device_info.is_cuda
        )

        self.step = 0
        self.tokens_seen = 0
        self.history: list[StepMetrics] = []
        self.eval_history: list[dict[str, float]] = []
        self.best_eval_loss = float("inf")
        self._batches: Iterator[dict[str, torch.Tensor]] | None = None

        if config.compile_model:
            # Off by default: on Windows torch.compile needs a working C++
            # toolchain, and a failure here would look like a training bug.
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                logger.info("model compiled")
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("torch.compile unavailable, running eager: %s", exc)

    # ---------------------------------------------------------------- internals

    def _autocast(self) -> Any:
        if self.dtype == torch.float32 or not self.device_info.is_cuda:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def _next_batch(self) -> dict[str, torch.Tensor]:
        if self._batches is None:
            self._batches = _cycle(self.train_loader)
        return next(self._batches)

    @staticmethod
    def _supervised(labels: torch.Tensor) -> int:
        """Tokens that actually contribute to the loss.

        For pretraining this is everything but the shifted-off token; for SFT it
        is only the assistant span. Reporting it separately is what makes an
        all-masked batch visible instead of mysterious.
        """
        return int((labels[:, 1:] != IGNORE_INDEX).sum().item())

    def train_step(self) -> StepMetrics:
        """One optimizer step, including all accumulation micro-steps."""
        self.model.train()
        started = time.perf_counter()
        learning_rate = self.schedule.set_step(self.step)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_tokens = 0
        total_supervised = 0

        for _ in range(self.config.grad_accum_steps):
            batch = batch_to_device(self._next_batch(), self.device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            with self._autocast():
                output = self.model(input_ids=input_ids, labels=labels)
                loss = output.loss

            if loss is None:
                raise RuntimeError("model returned no loss; labels were not passed")

            # Divide before backward so accumulated gradients are the mean over
            # the whole effective batch, not its sum.
            scaled = loss / self.config.grad_accum_steps
            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            total_loss += loss.detach().float().item()
            total_tokens += input_ids.numel()
            total_supervised += self._supervised(labels)

        if self.scaler.is_enabled():
            # Unscale first: a clip applied to scaled gradients would clip at the
            # wrong threshold.
            self.scaler.unscale_(self.optimizer)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        )

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        mean_loss = total_loss / self.config.grad_accum_steps
        self.tokens_seen += total_tokens
        self.step += 1

        metrics = StepMetrics(
            step=self.step,
            loss=mean_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            tokens=total_tokens,
            supervised_tokens=total_supervised,
            seconds=time.perf_counter() - started,
        )
        self.history.append(metrics)
        return metrics

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        """Mean loss over the held-out split.

        Weighted by supervised token count, not by batch, so a short final batch
        cannot skew the average.
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()
        limit = max_batches if max_batches is not None else self.config.eval_batches
        loss_sum = 0.0
        token_sum = 0
        batches = 0

        for batch in self.eval_loader:
            if limit is not None and batches >= limit:
                break
            batch = batch_to_device(batch, self.device)
            labels = batch["labels"]
            supervised = self._supervised(labels)
            if supervised == 0:
                continue
            with self._autocast():
                output = self.model(input_ids=batch["input_ids"], labels=labels)
            loss_sum += output.loss.detach().float().item() * supervised
            token_sum += supervised
            batches += 1

        self.model.train()
        if token_sum == 0:
            logger.warning("evaluation saw no supervised tokens")
            return {}

        mean_loss = loss_sum / token_sum
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError:
            perplexity = float("inf")
        result = {
            "loss": mean_loss,
            "perplexity": perplexity,
            "batches": float(batches),
            "tokens": float(token_sum),
            "step": float(self.step),
        }
        self.eval_history.append(result)
        self.best_eval_loss = min(self.best_eval_loss, mean_loss)
        return result

    def save(self, *, metrics: dict[str, float] | None = None) -> Path:
        """Write a checkpoint for the current step and update the run pointer."""
        directory = checkpoint_dir_for(self.config.run_dir, self.step)
        metadata = CheckpointMetadata(
            step=self.step,
            tokens_seen=self.tokens_seen,
            stage=self.config.stage,
            run_name=self.config.run_name,
            metrics=metrics or {},
            dataset=self.dataset_info,
            environment={
                "device": self.device_info.name,
                "precision": str(self.dtype).replace("torch.", ""),
                "grad_accum_steps": self.config.grad_accum_steps,
            },
        )
        # The compiled wrapper's state dict has a _orig_mod prefix; save the real
        # module so the checkpoint loads without torch.compile present.
        target = getattr(self.model, "_orig_mod", self.model)
        path = save_checkpoint(
            directory,
            target,
            metadata=metadata,
            optimizer=self.optimizer,
            scheduler=self.schedule,
            tokenizer=self.tokenizer,
        )
        write_latest_pointer(self.config.run_dir, path)
        prune_checkpoints(self.config.run_dir, self.config.keep_checkpoints)
        return path

    def load_training_state(self, state: dict[str, Any]) -> None:
        """Resume optimizer, schedule, RNG and counters from a checkpoint."""
        from bravien.training.checkpoint import restore_training_state

        self.step = restore_training_state(state, self.optimizer, self.schedule)
        self.tokens_seen = int(state.get("tokens_seen", 0))
        logger.info("resumed at step %d (%s tokens seen)", self.step, format_count(self.tokens_seen))

    # --------------------------------------------------------------------- loop

    def train(self) -> dict[str, Any]:
        """Run until `max_steps`. Returns a summary of what happened."""
        remaining = self.config.max_steps - self.step
        if remaining <= 0:
            logger.info("already at step %d, nothing to do", self.step)
            return self.summary()

        logger.info(
            "training %s: %d step(s), batch accum %d, %s, %s",
            self.config.run_name,
            remaining,
            self.config.grad_accum_steps,
            str(self.dtype).replace("torch.", ""),
            self.device_info.name,
        )

        stop_reason = "completed"
        last_saved_step = -1
        with Timer() as timer:
            while self.step < self.config.max_steps:
                metrics = self.train_step()

                if not math.isfinite(metrics.loss):
                    message = f"loss became {metrics.loss} at step {self.step}"
                    if self.config.stop_on_nan:
                        logger.error("%s; stopping", message)
                        stop_reason = "nan_loss"
                        break
                    logger.warning(message)

                if self.config.log_every and self.step % self.config.log_every == 0:
                    logger.info(metrics.format())

                if (
                    self.config.eval_every
                    and self.eval_loader is not None
                    and self.step % self.config.eval_every == 0
                ):
                    result = self.evaluate()
                    if result:
                        logger.info(
                            "step %6d  eval loss %.4f  ppl %.2f  (%d tokens)",
                            self.step,
                            result["loss"],
                            result["perplexity"],
                            int(result["tokens"]),
                        )

                if self.config.save_every and self.step % self.config.save_every == 0:
                    self.save(metrics={"train_loss": metrics.loss})
                    last_saved_step = self.step

        # Only if the periodic save did not already cover this step — rewriting an
        # identical checkpoint costs the full weight size in I/O.
        if (
            self.config.save_at_end
            and stop_reason == "completed"
            and last_saved_step != self.step
        ):
            final = self.evaluate() if self.eval_loader is not None else {}
            self.save(
                metrics={
                    "train_loss": self.history[-1].loss if self.history else float("nan"),
                    **({"eval_loss": final["loss"]} if final else {}),
                }
            )

        summary = self.summary()
        summary["stop_reason"] = stop_reason
        summary["wall_seconds"] = timer.elapsed
        logger.info(
            "finished %s in %s: %s",
            self.config.run_name,
            format_duration(timer.elapsed),
            stop_reason,
        )
        return summary

    def summary(self) -> dict[str, Any]:
        first = self.history[0].loss if self.history else None
        last = self.history[-1].loss if self.history else None
        recent = self.history[-20:]
        return {
            "run_name": self.config.run_name,
            "stage": self.config.stage,
            "steps": self.step,
            "tokens_seen": self.tokens_seen,
            "first_loss": first,
            "final_loss": last,
            "mean_recent_loss": (
                sum(m.loss for m in recent) / len(recent) if recent else None
            ),
            "best_eval_loss": (
                self.best_eval_loss if math.isfinite(self.best_eval_loss) else None
            ),
            "tokens_per_second": (
                sum(m.tokens for m in recent) / sum(m.seconds for m in recent)
                if recent and sum(m.seconds for m in recent) > 0
                else None
            ),
            "device": self.device_info.name,
            "precision": str(self.dtype).replace("torch.", ""),
            "run_dir": str(self.config.run_dir),
        }


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
