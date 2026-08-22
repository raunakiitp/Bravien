"""Bravien — an independent, locally-trainable language model.

The package is layered so each stage can be used on its own:

    bravien.model       architecture (config, transformer, generation)
    bravien.tokenizer   byte-level BPE training, encoding, chat templates
    bravien.data        corpus cleaning, filtering, dedup, tokenising, packing
    bravien.training    pretraining, instruction tuning, checkpoints
    bravien.evaluation  perplexity and behavioural test suites
    bravien.inference   KV-cached engine, sampling, local API server
    bravien.utils       hardware detection, seeding, logging
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
