"""Native Pretraining Engine for Bravien Transformer Architecture."""

from __future__ import annotations

import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bravien.model.bravien_config import BravienConfig
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.parameter_count import count_parameters
from bravien.training.checkpoint_manager import CheckpointManager, TrainingState
from bravien.training.pretrain_config import PretrainConfig


class SyntheticTokenDataset(Dataset):
    """Synthetic dataset for tiny smoke tests and training sanity checks."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Generate predictable repeating pattern for loss reduction verification
        pattern = [(idx + i) % (self.vocab_size - 10) + 5 for i in range(self.seq_len)]
        tokens = torch.tensor(pattern, dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def create_optimizer(model: BravienForCausalLM, config: PretrainConfig) -> torch.optim.AdamW:
    """Create AdamW optimizer with decoupled weight decay (exempting 1D norms and biases)."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate decay with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BravienPretrainer:
    """Production-grade native pretraining loop for Bravien models."""

    def __init__(
        self,
        model: BravienForCausalLM,
        config: PretrainConfig,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)

        # Optimizer & Scheduler
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_learning_rate / config.learning_rate,
        )

        # Mixed precision setup
        self.use_amp = self.device.type == "cuda" and config.mixed_precision in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp and self.amp_dtype == torch.float16 else None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.output_dir, keep_last_n=config.keep_last_n_checkpoints
        )

        # Graceful interruption handler
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        print("\n[BravienPretrainer] Interruption signal received. Saving checkpoint and exiting gracefully...")
        self.interrupted = True

    def train(self) -> dict[str, Any]:
        """Execute pretraining loop."""
        torch.manual_seed(self.config.seed)

        # Fallback to synthetic dataset if none provided (smoke mode)
        if self.train_dataset is None:
            self.train_dataset = SyntheticTokenDataset(
                vocab_size=self.model.config.vocab_size,
                seq_len=min(self.config.max_seq_len, self.model.config.max_position_embeddings),
                num_samples=1000,
            )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.micro_batch_size,
                shuffle=False,
            )

        print("\n" + "=" * 65)
        print(f"BRAVIEN PRETRAINING: {self.model.config.name}")
        print("=" * 65)
        param_report = count_parameters(self.model)
        print(f"Target Architecture:    {self.model.config.name} ({param_report['total_millions']:.2f}M params)")
        print(f"Device:                 {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")
        print(f"Mixed Precision:        {self.config.mixed_precision} (AMP: {self.use_amp})")
        print(f"Effective Batch Size:   {self.config.effective_batch_size} (micro: {self.config.micro_batch_size}, accum: {self.config.gradient_accumulation_steps})")
        print(f"Max Context Tokens:     {self.config.max_seq_len}")
        print(f"Total Target Steps:     {self.config.max_steps}")
        print(f"Checkpoint Directory:   {self.config.output_dir}")
        print("=" * 65 + "\n")

        self.model.train()
        step = self.config.initial_step
        epoch = 0
        total_tokens = self.config.initial_tokens
        running_loss = 0.0
        best_val_loss = float("inf")
        start_time = time.perf_counter()
        last_log_time = start_time
        tokens_since_log = 0
        last_saved_path: Path | None = None
        avg_loss = 0.0

        train_iter = iter(train_loader)

        target_max_steps = self.config.max_steps
        if self.config.initial_step > 0 and self.config.max_steps <= self.config.initial_step:
            target_max_steps = self.config.initial_step + self.config.max_steps

        while step < target_max_steps and not self.interrupted:
            if self.config.max_tokens is not None and total_tokens >= self.config.max_tokens:
                print(f"\n[TOKEN BUDGET REACHED] Reached requested token budget of {self.config.max_tokens:,} tokens (processed {total_tokens:,}). Stopping cleanly.")
                break

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    epoch += 1
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                num_tokens_in_batch = input_ids.numel()
                total_tokens += num_tokens_in_batch
                tokens_since_log += num_tokens_in_batch

                # Forward pass with AMP
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        output = self.model(input_ids=input_ids, labels=labels)
                        loss = output.loss / self.config.gradient_accumulation_steps
                else:
                    output = self.model(input_ids=input_ids, labels=labels)
                    loss = output.loss / self.config.gradient_accumulation_steps

                accum_loss += loss.item()

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Gradient clipping & Optimizer Step
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            step += 1
            running_loss += accum_loss

            # Periodic logging
            if step % self.config.log_interval == 0 or step == self.config.initial_step + 1:
                now = time.perf_counter()
                elapsed_since_log = now - last_log_time
                tps = tokens_since_log / elapsed_since_log if elapsed_since_log > 0 else 0
                avg_loss = running_loss / (self.config.log_interval if step > self.config.initial_step + 1 else 1)
                lr = self.scheduler.get_last_lr()[0]

                vram_str = ""
                if self.device.type == "cuda":
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    vram_str = f" | VRAM: {vram_mb:.0f}MB"

                print(
                    f"Step {step:05d}/{target_max_steps:05d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"GradNorm: {grad_norm:.2f} | "
                    f"Speed: {tps:.1f} tok/s"
                    f"{vram_str}"
                )
                running_loss = 0.0
                tokens_since_log = 0
                last_log_time = now

            # Periodic checkpoint saving
            if step % self.config.save_interval == 0 or step == target_max_steps:
                state = TrainingState(
                    step=step,
                    epoch=epoch,
                    best_loss=accum_loss,
                    total_tokens_trained=total_tokens,
                    elapsed_seconds=time.perf_counter() - start_time,
                )
                last_saved_path = self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    state=state,
                )
                print(f"  [CHECKPOINT] Saved checkpoint at step {step} -> {last_saved_path}")

        total_time = max(time.perf_counter() - start_time, 1e-6)
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if self.device.type == "cuda" else 0.0
        overall_tps = (total_tokens - self.config.initial_tokens) / total_time

        print(f"\nPretraining finished in {total_time:.2f}s! Total Tokens Processed: {total_tokens:,}")
        return {
            "final_step": step,
            "total_tokens": total_tokens,
            "tokens_trained_in_session": total_tokens - self.config.initial_tokens,
            "elapsed_seconds": total_time,
            "final_loss": accum_loss,
            "average_recent_loss": avg_loss or accum_loss,
            "tokens_per_second": overall_tps,
            "peak_gpu_memory_mb": peak_vram_mb,
            "last_checkpoint_path": str(last_saved_path) if last_saved_path else None,
        }
