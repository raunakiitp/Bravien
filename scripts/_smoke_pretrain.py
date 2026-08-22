"""Manual pretraining smoke run against the prepared synthetic corpus.

Not a test — a real short run whose only purpose is to prove the loop, the
checkpoint, and reloading all work on this machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from bravien.model.config import BravienConfig
from bravien.training.checkpoint import load_checkpoint, resolve_latest
from bravien.training.optimizer import OptimizerConfig, parameter_group_summary
from bravien.training.pretrain import PretrainRun, run_pretraining
from bravien.training.scheduler import SchedulerConfig
from bravien.training.trainer import TrainingConfig
from bravien.utils.logging import configure_stdout, setup_logging

configure_stdout()
setup_logging(level="INFO")

TOKEN_PATH = REPO_ROOT / "datasets" / "smoke" / "smoke.bin"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "smoke-tokenizer"

model_config = BravienConfig(
    name="bravien-smoke",
    vocab_size=1021,          # overridden from the tokenizer anyway
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=688,
    max_position_embeddings=256,
    dropout=0.0,
)

training = TrainingConfig(
    run_name="smoke-pretrain",
    stage="pretrain",
    max_steps=200,
    grad_accum_steps=1,
    optimizer=OptimizerConfig(learning_rate=6e-4, weight_decay=0.1),
    scheduler=SchedulerConfig(warmup_steps=20, total_steps=200, min_lr_ratio=0.1),
    log_every=20,
    eval_every=50,
    eval_batches=8,
    save_every=100,
    keep_checkpoints=2,
    seed=0,
)

run = PretrainRun(
    token_path=TOKEN_PATH,
    tokenizer_path=TOKENIZER_PATH,
    model=model_config,
    training=training,
    batch_size=16,
    seq_len=255,
    num_workers=0,
    resume=False,
)

summary = run_pretraining(run)
print("\n=== summary ===")
print(json.dumps(summary, indent=2, default=str))

# --- reload the checkpoint and generate, with no trainer in scope -------------
latest = resolve_latest(training.run_dir)
print(f"\nlatest checkpoint: {latest}")
assert latest is not None

loaded = load_checkpoint(latest, device="cpu", load_tokenizer=True)
print(f"loaded step {loaded.metadata.step}, params {loaded.metadata.parameters:,}")
print(f"bundled tokenizer: {loaded.tokenizer}")
print(f"metadata.metrics: {loaded.metadata.metrics}")

tok = loaded.tokenizer
assert tok is not None
prompt = "Notes on tides.\n\nThe engineer measured"
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
loaded.model.eval()
with torch.no_grad():
    out = loaded.model.generate(ids, max_new_tokens=40, temperature=0.8, top_k=40, seed=0)
print("\n--- generation from the reloaded checkpoint ---")
print(tok.decode(out[0].tolist()))

# --- bit-identical reload check ---------------------------------------------
second = load_checkpoint(latest, device="cpu", load_tokenizer=False)
identical = all(
    torch.equal(a, b)
    for a, b in zip(
        loaded.model.state_dict().values(), second.model.state_dict().values(), strict=True
    )
)
print(f"\ntwo loads bit-identical: {identical}")
