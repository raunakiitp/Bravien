"""Data pipeline tests: instruction parsing, packing, batching (§64).

Most of these guard against a failure mode that does not raise: data that is
silently wrong. A conversation whose roles were misread, a batch whose padding is
counted as supervision, an instruction file that parsed to nothing — each of those
trains happily and produces a worse model with no error to point at.
"""

from __future__ import annotations

import json

import pytest
import torch

from bravien.data.instructions import (
    SEED_SYSTEM_PROMPT,
    InstructionDataError,
    LoadStats,
    conversation_from_record,
    generate_seed_conversations,
    load_conversations_jsonl,
    split_conversations,
    validate_conversation,
)
from bravien.data.pack import (
    SFTExample,
    SFTPackStats,
    build_sft_example,
    build_sft_examples,
    collate_sft,
    count_packed_sequences,
    pack_sequences,
)
from bravien.tokenizer.tokenizer import IGNORE_INDEX

# ------------------------------------------------------- instruction file shapes


def test_reads_the_messages_shape():
    record = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    assert conversation_from_record(record) == record["messages"]


def test_reads_the_sharegpt_shape():
    """`from`/`value` with human/gpt role names, as distributed datasets use."""
    record = {
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
    }
    assert conversation_from_record(record) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_reads_a_flat_prompt_and_reply_pair():
    record = {"instruction": "Summarise this.", "output": "Done."}
    assert conversation_from_record(record) == [
        {"role": "user", "content": "Summarise this."},
        {"role": "assistant", "content": "Done."},
    ]


def test_alpaca_input_is_appended_to_the_instruction():
    """Dropping `input` would leave the reply unexplainable by its prompt."""
    record = {"instruction": "Translate.", "input": "bonjour", "output": "hello"}
    conversation = conversation_from_record(record)
    assert "Translate." in conversation[0]["content"]
    assert "bonjour" in conversation[0]["content"]


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"instruction": "no reply here"},
        {"output": "no prompt here"},
        {"messages": []},
        {"nothing": "recognisable"},
    ],
)
def test_unrecognisable_records_are_rejected_not_guessed_at(record):
    assert conversation_from_record(record) is None


def test_a_conversation_must_end_with_the_assistant():
    """The final turn is the supervision; a trailing user turn has no target."""
    with pytest.raises(InstructionDataError):
        validate_conversation(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )


def test_a_conversation_without_an_assistant_turn_is_rejected():
    with pytest.raises(InstructionDataError):
        validate_conversation([{"role": "user", "content": "a"}])


# ------------------------------------------------------------------ file loading


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r) if not isinstance(r, str) else r for r in records),
        encoding="utf-8",
    )
    return path


def test_loads_conversations_and_counts_what_it_skipped(tmp_path):
    path = _write_jsonl(
        tmp_path / "data.jsonl",
        [
            {"instruction": "a", "output": "b"},
            "",
            "{not json",
            {"unrecognised": True},
            {"messages": [{"role": "user", "content": "x"}]},
            {"instruction": "c", "output": "d"},
        ],
    )
    stats = LoadStats()
    conversations = load_conversations_jsonl(path, stats=stats)

    assert len(conversations) == 2
    assert stats.kept == 2
    assert stats.blank == 1
    assert stats.bad_json == 1
    assert stats.unrecognised == 1
    # A user-only conversation parses, but carries no supervision. The loader
    # files that under `empty_content` — the assistant said nothing usable —
    # keeping `invalid_roles` for records whose role names were unreadable.
    assert stats.empty_content == 1
    assert stats.invalid_roles == 0
    assert stats.dropped == 4
    assert "2 kept" in stats.format()


def test_a_file_that_parses_to_nothing_is_an_error(tmp_path):
    """Training on an empty dataset would otherwise look like a successful run."""
    path = _write_jsonl(tmp_path / "junk.jsonl", ["{bad", "{also bad"])
    with pytest.raises(InstructionDataError):
        load_conversations_jsonl(path)


def test_limit_stops_reading_early(tmp_path):
    path = _write_jsonl(
        tmp_path / "data.jsonl", [{"instruction": str(i), "output": "x"} for i in range(50)]
    )
    assert len(load_conversations_jsonl(path, limit=7)) == 7


def test_a_missing_file_is_reported_as_such(tmp_path):
    with pytest.raises((FileNotFoundError, InstructionDataError)):
        load_conversations_jsonl(tmp_path / "absent.jsonl")


# ------------------------------------------------------- generated conversations


def test_seed_conversations_are_well_formed():
    conversations = list(generate_seed_conversations(200, seed=0))
    assert len(conversations) == 200
    for conversation in conversations:
        assert conversation[0]["role"] == "system"
        assert conversation[0]["content"] == SEED_SYSTEM_PROMPT
        assert conversation[-1]["role"] == "assistant"
        # Roles alternate after the system turn.
        roles = [m["role"] for m in conversation[1:]]
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        assert all(m["content"].strip() for m in conversation)
        validate_conversation(conversation)


def test_seed_conversations_are_reproducible():
    assert list(generate_seed_conversations(30, seed=7)) == list(
        generate_seed_conversations(30, seed=7)
    )


def test_a_different_seed_gives_different_conversations():
    assert list(generate_seed_conversations(30, seed=1)) != list(
        generate_seed_conversations(30, seed=2)
    )


def test_the_system_prompt_can_be_omitted():
    conversation = list(generate_seed_conversations(5, seed=0, system_prompt=None))[0]
    assert conversation[0]["role"] == "user"


def test_identity_answers_never_claim_to_be_another_product():
    """§72: the model must not be trained to introduce itself as something else."""
    replies = " ".join(
        m["content"]
        for c in generate_seed_conversations(600, seed=0, identity_share=1.0)
        for m in c
        if m["role"] == "assistant"
    ).lower()
    assert "bravien" in replies
    for foreign in ("i am chatgpt", "i am claude", "i am gpt-4", "as an openai"):
        assert foreign not in replies


# ------------------------------------------------------------------ splitting


def test_split_partitions_the_input_exactly():
    """Every conversation lands on exactly one side, and none is invented.

    Compared as a multiset rather than a set: the generated corpus repeats short
    identity answers on purpose, so the same conversation legitimately appears
    more than once and set-based disjointness would be the wrong property.
    """
    from collections import Counter

    conversations = list(generate_seed_conversations(300, seed=0))
    split = split_conversations(conversations, val_fraction=0.1, seed=0)

    assert len(split.train) + len(split.validation) == 300
    assert len(split.validation) == 30

    counted = Counter(json.dumps(c) for c in split.train)
    counted.update(json.dumps(c) for c in split.validation)
    assert counted == Counter(json.dumps(c) for c in conversations)


def test_min_val_protects_a_tiny_dataset():
    """A 5% split of 40 examples is 2, which measures nothing."""
    split = split_conversations(
        list(generate_seed_conversations(40, seed=0)), val_fraction=0.05, min_val=8
    )
    assert len(split.validation) == 8


def test_split_leaves_training_data_behind_even_when_asked_not_to():
    split = split_conversations(
        list(generate_seed_conversations(10, seed=0)), val_fraction=0.99, min_val=8
    )
    assert split.train


def test_split_is_reproducible():
    conversations = list(generate_seed_conversations(100, seed=0))
    first = split_conversations(conversations, seed=3)
    second = split_conversations(conversations, seed=3)
    assert first.validation == second.validation


# -------------------------------------------------------------------- packing


def _conversation():
    return [
        {"role": "user", "content": "What is the reading?"},
        {"role": "assistant", "content": "6 minutes."},
    ]


def test_a_packed_example_supervises_only_the_reply(trained_tokenizer):
    example = build_sft_example(_conversation(), trained_tokenizer, max_length=128)
    assert example is not None
    assert 0 < example.num_supervised < len(example.input_ids)
    trainable = [
        label for label in example.labels if label != IGNORE_INDEX
    ]
    assert "6 minutes." in trained_tokenizer.decode(
        trainable, skip_special_tokens=False
    )


def test_truncation_keeps_the_answer_and_drops_old_context(trained_tokenizer):
    """Left truncation: the reply is the supervision, so it must survive."""
    conversation = [
        {"role": "user", "content": "context " * 400},
        {"role": "assistant", "content": "the answer"},
    ]
    example = build_sft_example(conversation, trained_tokenizer, max_length=64)
    assert example is not None
    assert len(example.input_ids) == 64
    assert example.num_supervised > 0


def test_a_conversation_with_no_reply_produces_no_example(trained_tokenizer):
    """No assistant turn means no supervision, so there is nothing to train on."""
    user_only = [{"role": "user", "content": "no reply follows"}]
    assert build_sft_example(user_only, trained_tokenizer, max_length=128) is None


def test_an_example_supervising_only_the_stop_marker_is_dropped(trained_tokenizer):
    """A window too small to reach the reply teaches "always stop immediately".

    The trailing `<EOS>` is supervised and always last, so such an example does
    have a target — which is exactly why it has to be excluded explicitly rather
    than caught by an all-masked check.
    """
    conversation = [
        {"role": "user", "content": "context " * 300},
        {"role": "assistant", "content": "the answer"},
    ]
    assert build_sft_example(conversation, trained_tokenizer, max_length=2) is None


def test_left_truncation_keeps_the_answer_when_the_window_allows(trained_tokenizer):
    conversation = [
        {"role": "user", "content": "context " * 300},
        {"role": "assistant", "content": "the answer"},
    ]
    for max_length in (16, 32, 256):
        example = build_sft_example(
            conversation, trained_tokenizer, max_length=max_length
        )
        assert example is not None, max_length
        assert example.num_supervised > 0, max_length


def test_pack_stats_add_up(trained_tokenizer):
    stats = SFTPackStats()
    examples = list(
        build_sft_examples(
            [_conversation() for _ in range(12)],
            trained_tokenizer,
            max_length=128,
            stats=stats,
        )
    )
    assert stats.conversations == 12
    assert stats.kept == len(examples) == 12
    assert stats.supervised_tokens == sum(e.num_supervised for e in examples)
    assert 0 < stats.supervision_rate < 1


def test_pack_sequences_produces_fixed_length_blocks():
    blocks = list(pack_sequences(range(25), seq_len=10))
    assert blocks == [list(range(10)), list(range(10, 20))]
    assert count_packed_sequences(25, 10) == 2


def test_pack_sequences_can_keep_a_short_tail():
    blocks = list(pack_sequences(range(25), seq_len=10, drop_last=False))
    assert len(blocks) == 3
    assert blocks[-1] == list(range(20, 25))


# ------------------------------------------------------------------- collation


def test_collate_pads_on_the_right_and_masks_the_padding():
    batch = [
        SFTExample(input_ids=[5, 6, 7], labels=[IGNORE_INDEX, 6, 7]),
        SFTExample(input_ids=[8], labels=[8]),
    ]
    collated = collate_sft(batch, pad_id=0, pad_to_multiple_of=1)

    assert collated["input_ids"].shape == (2, 3)
    # Right padding is what makes the attention mask unnecessary under causal
    # masking: a real token at position i only ever attends to j <= i.
    assert collated["input_ids"][1].tolist() == [8, 0, 0]
    assert collated["labels"][1].tolist() == [8, IGNORE_INDEX, IGNORE_INDEX]
    assert collated["attention_mask"][1].tolist() == [1, 0, 0]


def test_collate_rounds_the_sequence_length_up():
    batch = [SFTExample(input_ids=[1, 2, 3], labels=[1, 2, 3])]
    collated = collate_sft(batch, pad_id=0, pad_to_multiple_of=8)
    assert collated["input_ids"].shape == (1, 8)


def test_collated_padding_is_never_counted_as_supervision():
    batch = [
        SFTExample(input_ids=[1, 2, 3, 4], labels=[1, 2, 3, 4]),
        SFTExample(input_ids=[9], labels=[9]),
    ]
    collated = collate_sft(batch, pad_id=0, pad_to_multiple_of=1)
    supervised = (collated["labels"] != IGNORE_INDEX).sum().item()
    assert supervised == 5
    assert torch.equal(
        collated["labels"] != IGNORE_INDEX, collated["attention_mask"].bool()
    )
