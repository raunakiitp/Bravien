"""HuggingFace Model SFT Fine-Tuning Engine for Bravien Stage 3.

Provides memory-safe, high-throughput, resumable supervised fine-tuning
specifically designed for causal language models like Qwen/Qwen2.5-0.5B-Instruct
on modern GPUs (e.g. NVIDIA RTX 4050 6GB) with bfloat16 mixed precision.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bravien.data.schema import BravienTrainingExample
from bravien.utils.hardware import DeviceInfo, detect_device
from bravien.utils.logging import format_count, format_duration, get_logger
from bravien.utils.paths import ensure_dir

logger = get_logger("training.hf_trainer")


@dataclass
class HFTrainingConfig:
    """Hyperparameters and runtime settings for Bravien model fine-tuning."""

    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: Path = Path("checkpoints/bravien-v1")
    train_data_path: Path = Path("data/processed/train.jsonl")
    val_data_path: Path = Path("data/processed/val.jsonl")
    test_data_path: Path = Path("data/processed/test.jsonl")

    max_steps: int = 400
    batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 40
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 512

    precision: str = "bfloat16"  # "bfloat16", "float16", "float32"
    device: str | None = None
    seed: int = 42

    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    eval_batches: int = 20

    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.train_data_path = Path(self.train_data_path)
        self.val_data_path = Path(self.val_data_path)
        self.test_data_path = Path(self.test_data_path)


class SFTDataset(Dataset):
    """PyTorch Dataset that encodes multi-turn conversations with assistant-only loss masking."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        limit: int | None = None,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    self.examples.append(record)
                    if limit and len(self.examples) >= limit:
                        break
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.examples[idx]
        messages = item.get("messages", [])

        # Format conversation and compute assistant token masks
        input_ids: list[int] = []
        labels: list[int] = []

        # Use tokenizer chat template formatting or manual Qwen format
        for m in messages:
            role = m["role"]
            content = m["content"]
            # Qwen format: <|im_start|>role\ncontent<|im_end|>\n
            role_header = f"<|im_start|>{role}\n"
            content_str = f"{content}<|im_end|>\n"

            header_ids = self.tokenizer.encode(role_header, add_special_tokens=False)
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)

            segment_ids = header_ids + content_ids

            input_ids.extend(segment_ids)
            if role == "assistant":
                # Mask header, train only on assistant reply tokens
                labels.extend([-100] * len(header_ids))
                labels.extend(content_ids)
            else:
                # Mask system and user turns completely (-100)
                labels.extend([-100] * len(segment_ids))

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def sft_collate_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad batch sequences to the maximum length in the current batch."""
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    batch_attn_masks: list[torch.Tensor] = []

    for item in batch:
        inp = item["input_ids"]
        lab = item["labels"]
        pad_len = max_len - len(inp)

        if pad_len > 0:
            padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            padded_lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            padded_attn = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        else:
            padded_inp = inp
            padded_lab = lab
            padded_attn = item["attention_mask"]

        batch_input_ids.append(padded_inp)
        batch_labels.append(padded_lab)
        batch_attn_masks.append(padded_attn)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attn_masks),
    }


class HFTrainer:
    """Orchestrates model loading, training loop, validation, metrics, and checkpointing."""

    def __init__(self, config: HFTrainingConfig | None = None) -> None:
        self.config = config or HFTrainingConfig()
        self.device_info = detect_device(self.config.device)
        self.device = torch.device(self.device_info.device)

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # Set torch precision
        if self.config.precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.torch_dtype = torch.bfloat16
        elif self.config.precision == "float16" and torch.cuda.is_available():
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        logger.info(f"Loading base model {self.config.base_model} on {self.device_info.name} ({self.torch_dtype})...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(self.device)

        # Prepare optimizer
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.history: list[dict[str, Any]] = []

    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / max(1, self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.config.min_lr + (self.config.learning_rate - self.config.min_lr) * cosine

    def evaluate(self, val_loader: DataLoader) -> float:
        """Run validation loss evaluation."""
        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= self.config.eval_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=self.torch_dtype, enabled=self.device.type == "cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    batches += 1

        self.model.train()
        return total_loss / max(1, batches)

    def train(self) -> dict[str, Any]:
        """Execute the full SFT training loop."""
        ensure_dir(self.config.output_dir)

        train_dataset = SFTDataset(
            self.config.train_data_path,
            self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        val_dataset = SFTDataset(
            self.config.val_data_path,
            self.tokenizer,
            max_length=self.config.max_seq_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: sft_collate_fn(b, self.tokenizer.pad_token_id or 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: sft_collate_fn(b, self.tokenizer.pad_token_id or 0),
        )

        logger.info(
            f"Starting Bravien SFT: {len(train_dataset):,} train examples, "
            f"{len(val_dataset):,} val examples, max_steps={self.config.max_steps}, "
            f"batch_size={self.config.batch_size}x{self.config.grad_accum_steps} (eff={self.config.batch_size * self.config.grad_accum_steps})"
        )

        self.model.train()
        self.optimizer.zero_grad()

        start_time = time.perf_counter()
        accum_loss = 0.0
        total_tokens_seen = 0
        total_examples_seen = 0

        train_iter = iter(train_loader)

        while self.step < self.config.max_steps:
            # Accumulate gradients across grad_accum_steps
            step_loss = 0.0
            for _ in range(self.config.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                total_tokens_seen += int(attention_mask.sum().item())
                total_examples_seen += len(input_ids)

                with torch.autocast(device_type=self.device.type, dtype=self.torch_dtype, enabled=self.device.type == "cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.config.grad_accum_steps

                loss.backward()
                step_loss += loss.item()

            # Gradient clipping & Optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            lr = self.get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.step += 1
            accum_loss += step_loss

            # Logging
            if self.step % self.config.log_every == 0 or self.step == 1:
                avg_train_loss = accum_loss / (self.config.log_every if self.step > 1 else 1)
                accum_loss = 0.0
                elapsed = time.perf_counter() - start_time
                tok_per_sec = total_tokens_seen / max(0.1, elapsed)

                vram_info = ""
                if torch.cuda.is_available():
                    alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    vram_info = f" | VRAM: {alloc_mb:.0f}MB"

                logger.info(
                    f"Step {self.step:4d}/{self.config.max_steps} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"{tok_per_sec:.0f} tok/s | "
                    f"{format_duration(elapsed)}{vram_info}"
                )

                self.history.append({
                    "step": self.step,
                    "train_loss": avg_train_loss,
                    "lr": lr,
                    "tokens_seen": total_tokens_seen,
                    "examples_seen": total_examples_seen,
                    "elapsed_seconds": elapsed,
                })

            # Validation
            if self.step % self.config.eval_every == 0 or self.step == self.config.max_steps:
                val_loss = self.evaluate(val_loader)
                logger.info(f"==> Validation at Step {self.step}: Val Loss = {val_loss:.4f} (Best: {self.best_val_loss:.4f})")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_dir = self.config.output_dir / "best"
                    ensure_dir(best_dir)
                    self.save_checkpoint(best_dir)

            # Periodic checkpoint save
            if self.step % self.config.save_every == 0:
                step_dir = self.config.output_dir / f"step-{self.step:05d}"
                ensure_dir(step_dir)
                self.save_checkpoint(step_dir)

        # Final Checkpoint Save
        final_dir = self.config.output_dir
        self.save_checkpoint(final_dir)
        total_time = time.perf_counter() - start_time

        logger.info(
            f"Bravien SFT completed successfully: {self.step} steps, "
            f"best_val_loss={self.best_val_loss:.4f} in {format_duration(total_time)}"
        )

        return {
            "status": "completed",
            "steps": self.step,
            "best_val_loss": self.best_val_loss,
            "total_tokens_seen": total_tokens_seen,
            "total_examples_seen": total_examples_seen,
            "elapsed_seconds": total_time,
            "checkpoint_path": str(final_dir),
        }

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """Save model weights, tokenizer, and training metadata."""
        ensure_dir(checkpoint_dir)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = {
            "model_name": "bravien-v1",
            "base_model": self.config.base_model,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "timestamp": time.time(),
            "config": asdict(self.config),
        }
        with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        with open(checkpoint_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
