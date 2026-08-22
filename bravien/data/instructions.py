"""Instruction data for supervised fine-tuning (§18, §26).

Pretraining teaches Bravien what its language looks like. It does not teach the
model that a turn from a user is a *request*, that a reply belongs between
`<ASSISTANT>` markers, or that a reply should end. Those come from conversations,
and this module is where conversations come from.

Two sources, in priority order:

* **A local JSONL file** you supply. Several common layouts are accepted, because
  instruction sets in the wild disagree about field names and rejecting a corpus
  over a key called `output` instead of `response` wastes everyone's time.
* **A generated seed set**, when there is no file. It is built from the same
  lexicon as `generate_seed_corpus`, so the fine-tuning distribution sits inside
  the pretraining one — the model is being taught a *format*, not a new language,
  which is the only thing a model this small can pick up in a few hundred steps.

On the seed set and honesty (§52, §71): these are training targets, not canned
answers. Nothing here is ever returned to a user. After fine-tuning, the model's
reply to any of these prompts is whatever its weights produce — often not the
target, because a 3M-parameter model fits its data badly. The identity examples
are included precisely so the model is not left to invent a claim about what it
is; whether it learned them is an evaluation question, not an assumption.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bravien.data.manifest import SourceRecord
from bravien.tokenizer.templates import ChatTemplateError, normalize_messages
from bravien.utils.logging import get_logger

logger = get_logger("data.instructions")

#: One conversation: an ordered list of `{"role": ..., "content": ...}`.
Conversation = list[dict[str, str]]

#: Field names seen in the wild for the prompt and the reply, in the order they
#: are tried. `instruction`/`output` is Alpaca's; `prompt`/`response` is common
#: in preference data; `question`/`answer` in QA sets.
_PROMPT_KEYS = ("instruction", "prompt", "question", "input", "user")
_REPLY_KEYS = ("output", "response", "answer", "completion", "assistant")


class InstructionDataError(ValueError):
    pass


@dataclass
class LoadStats:
    """What survived reading a file, and why the rest did not."""

    lines: int = 0
    kept: int = 0
    blank: int = 0
    bad_json: int = 0
    unrecognised: int = 0
    invalid_roles: int = 0
    empty_content: int = 0

    @property
    def dropped(self) -> int:
        return self.blank + self.bad_json + self.unrecognised + self.invalid_roles + self.empty_content

    def format(self) -> str:
        parts = [f"{self.kept:,} kept of {self.lines:,} lines"]
        for name in ("blank", "bad_json", "unrecognised", "invalid_roles", "empty_content"):
            count = getattr(self, name)
            if count:
                parts.append(f"{count:,} {name}")
        return ", ".join(parts)


def _first_present(record: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def conversation_from_record(record: dict[str, Any]) -> Conversation | None:
    """Coerce one JSON object into a conversation, or None if it is not one.

    Handles three shapes:

    1. ``{"messages": [{"role": ..., "content": ...}, ...]}`` — used as-is.
    2. ``{"conversations": [...]}`` with ``from``/``value`` keys, the ShareGPT
       layout, whose role names differ (``human``, ``gpt``).
    3. A flat prompt/reply pair under any of the aliases above, optionally with a
       ``system`` field and an Alpaca-style ``input`` appended to the prompt.
    """
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in messages
            if isinstance(m, dict)
        ]

    shared = record.get("conversations")
    if isinstance(shared, list) and shared:
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "chatgpt": "assistant",
            "system": "system",
            "tool": "tool",
        }
        out: Conversation = []
        for turn in shared:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("from", turn.get("role", ""))).lower()
            out.append(
                {
                    "role": role_map.get(speaker, speaker),
                    "content": str(turn.get("value", turn.get("content", ""))),
                }
            )
        return out or None

    prompt = _first_present(record, _PROMPT_KEYS)
    reply = _first_present(record, _REPLY_KEYS)
    if prompt is None or reply is None:
        return None

    # Alpaca splits the request across `instruction` and `input`. Joining them is
    # what makes the task well-posed: the instruction alone often has no answer.
    extra = record.get("input")
    if isinstance(extra, str) and extra.strip() and extra is not prompt:
        prompt = f"{prompt}\n\n{extra.strip()}"

    conversation: Conversation = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        conversation.append({"role": "system", "content": system})
    conversation.append({"role": "user", "content": prompt})
    conversation.append({"role": "assistant", "content": reply})
    return conversation


def validate_conversation(messages: Conversation) -> Conversation:
    """Check roles and require something for the model to be trained on.

    A conversation with no assistant turn carries no supervision at all: it would
    tokenize to an all-masked example, which `build_sft_example` drops later. It
    is cheaper and clearer to reject it here, where the reason can be named.
    """
    normalized = normalize_messages(messages)
    if not any(m["role"] == "assistant" and m["content"].strip() for m in normalized):
        raise InstructionDataError("no non-empty assistant turn")
    if normalized[-1]["role"] != "assistant":
        raise InstructionDataError(
            f"conversation ends with a {normalized[-1]['role']!r} turn; the last "
            "turn must be the assistant's, or there is nothing to learn from it"
        )
    return normalized


def load_conversations_jsonl(
    path: str | Path, *, stats: LoadStats | None = None, limit: int | None = None
) -> list[Conversation]:
    """Read conversations from a JSON Lines file.

    Malformed lines are counted and skipped rather than aborting the run: a single
    truncated record at the end of a 100k-line file should not cost a fine-tune.
    An empty result *is* an error, because it means the layout was not understood
    and training on nothing would otherwise look like success.
    """
    path = Path(path)
    if not path.is_file():
        raise InstructionDataError(f"no instruction file at {path}")

    st = stats if stats is not None else LoadStats()
    out: list[Conversation] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(out) >= limit:
                break
            st.lines += 1
            line = line.strip()
            if not line:
                st.blank += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                st.bad_json += 1
                continue
            if not isinstance(record, dict):
                st.unrecognised += 1
                continue

            conversation = conversation_from_record(record)
            if conversation is None:
                st.unrecognised += 1
                continue
            try:
                out.append(validate_conversation(conversation))
            except InstructionDataError:
                st.empty_content += 1
            except ChatTemplateError:
                st.invalid_roles += 1
            else:
                st.kept += 1

    if not out:
        raise InstructionDataError(
            f"{path} yielded no usable conversations ({st.format()}). Expected one "
            "JSON object per line with either a 'messages' list or a "
            f"prompt/reply pair under one of {list(_PROMPT_KEYS)} and "
            f"{list(_REPLY_KEYS)}."
        )

    logger.info("loaded %s from %s", st.format(), path)
    return out


def jsonl_record(path: str | Path, conversations: int) -> SourceRecord:
    """Provenance for a user-supplied instruction file."""
    path = Path(path)
    return SourceRecord(
        name=f"local-instructions:{path.name}",
        url=f"file://{path.resolve().as_posix()}",
        license="unknown (supplied locally; not distributed with this repository)",
        documents=conversations,
        bytes=path.stat().st_size if path.is_file() else 0,
        processing=["jsonl parse", "role validation"],
        notes="Instruction data provided by the operator. Bravien does not know "
        "its licence or provenance; record those yourself before publishing a "
        "model trained on it.",
    )


# ------------------------------------------------------------- generated seed set

#: Deliberately the vocabulary of `generate_seed_corpus`. Fine-tuning on words the
#: base model has never seen would spend the whole run learning embeddings.
_TOPICS = (
    "tides", "maps", "clocks", "bridges", "gardens", "letters", "harbours",
    "instruments", "notebooks", "telescopes", "presses", "archives",
)
_ITEMS = (
    "the tide charts", "a folding map", "the brass instrument", "two old letters",
    "the harbour lights", "a wooden model", "the printing press", "several drawings",
    "the weather log", "a small telescope", "the copper kettle", "three notebooks",
)
_PLACES = (
    "in the observatory", "beside the canal", "under the pier", "at the workshop",
    "near the old bridge", "inside the greenhouse", "across the courtyard",
    "along the coast road", "behind the market", "within the archive",
)
_UNITS = ("metres", "degrees", "minutes", "grams")

#: What Bravien is allowed to say about itself. Short, literal, and repeated often
#: enough that a small model has a chance of fitting it. Nothing here claims a
#: capability the code does not have (§71).
_IDENTITY: tuple[tuple[str, str], ...] = (
    (
        "What are you?",
        "I am Bravien, a small language model. My tokenizer, weights and "
        "inference engine were built in this repository and I run on this "
        "machine.",
    ),
    (
        "Who made you?",
        "I was trained from scratch in the Bravien repository. I am not a copy "
        "of another model and I do not call one.",
    ),
    (
        "Are you ChatGPT?",
        "No. I am Bravien. I am a much smaller model and I run locally.",
    ),
    (
        "Are you GPT-4?",
        "No. I am Bravien, a small model trained in this repository.",
    ),
    (
        "Are you Claude?",
        "No. I am Bravien, and I am not connected to any hosted model.",
    ),
    (
        "Do you need the internet?",
        "No. My weights are on this machine, so I keep answering with the "
        "network unplugged.",
    ),
    (
        "How big are you?",
        "I am small — a few million parameters. I make mistakes that a large "
        "model would not.",
    ),
    (
        "Can you run code?",
        "No. I cannot run code or commands on this machine.",
    ),
    (
        "Can you search the web?",
        "No. I answer from my weights alone.",
    ),
    (
        "Can you see images?",
        "No. I only read and write text.",
    ),
    (
        "What is your name?",
        "My name is Bravien.",
    ),
    (
        "Should I trust your answers?",
        "Check anything that matters. I am a small model and I am often wrong.",
    ),
)


def _copy_task(rng: random.Random) -> tuple[str, str]:
    """Repeat-back. Teaches attending to the prompt instead of the prior alone."""
    phrase = f"{rng.choice(_ITEMS)} {rng.choice(_PLACES)}"
    return f"Repeat this back exactly: {phrase}", phrase


def _topic_note(rng: random.Random) -> tuple[str, str]:
    """Produce corpus-shaped prose on request. The format the base model knows."""
    topic = rng.choice(_TOPICS)
    body = " ".join(
        f"{rng.choice(_ITEMS).capitalize()} was recorded {rng.choice(_PLACES)}."
        for _ in range(rng.randint(2, 3))
    )
    return f"Write a short note about {topic}.", f"Notes on {topic}. {body}"


def _list_task(rng: random.Random) -> tuple[str, str]:
    """Fixed-count listing. Teaches stopping after n items."""
    count = rng.randint(2, 3)
    picks = rng.sample(_TOPICS, count)
    listed = ", ".join(picks[:-1]) + f" and {picks[-1]}."
    return f"Name {count} topics from the archive.", listed


def _reading_task(rng: random.Random) -> tuple[str, str]:
    """Extract a number from the prompt. Answerable only by reading it."""
    value = rng.randint(1, 99)
    unit = rng.choice(_UNITS)
    day = rng.randint(1, 365)
    return (
        f"The reading was {value} {unit} on day {day}. What was the reading?",
        f"{value} {unit}.",
    )


def _where_task(rng: random.Random) -> tuple[str, str]:
    """Extract a place from the prompt."""
    item, place = rng.choice(_ITEMS), rng.choice(_PLACES)
    return (
        f"{item.capitalize()} was found {place}. Where was it found?",
        f"{place.capitalize()}.",
    )


def _unanswerable_task(rng: random.Random) -> tuple[str, str]:
    """Say so instead of inventing. The one habit most worth teaching (§72)."""
    topic = rng.choice(_TOPICS)
    return (
        f"What did the log say about {topic} on the missing page?",
        "I do not know. That is not in what I was given.",
    )


_TASK_BUILDERS = (
    _copy_task,
    _topic_note,
    _list_task,
    _reading_task,
    _where_task,
    _unanswerable_task,
)

#: Prepended to every generated conversation. The engine sends the same string at
#: inference, so keeping one definition here is what stops the two drifting.
SEED_SYSTEM_PROMPT = (
    "You are Bravien, a small language model running locally. Answer briefly and "
    "say when you do not know."
)


def generate_seed_conversations(
    count: int = 2000,
    *,
    seed: int = 0,
    system_prompt: str | None = SEED_SYSTEM_PROMPT,
    identity_share: float = 0.12,
) -> Iterator[Conversation]:
    """Synthesise instruction data, for use with no network.

    Args:
        identity_share: fraction of examples drawn from `_IDENTITY`. Oversampled
            relative to their number on purpose — there are a dozen of them and
            they are the examples whose failure is most visible to a user.

    A model fine-tuned only on this learns the *shape* of a reply and a handful of
    facts about itself. It does not become an assistant, and the manifest records
    it as synthetic so no evaluation can quietly present it as one (§71).
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= identity_share <= 1.0:
        raise ValueError("identity_share must be between 0 and 1")

    rng = random.Random(seed)
    for _ in range(count):
        if rng.random() < identity_share:
            prompt, reply = rng.choice(_IDENTITY)
        else:
            prompt, reply = rng.choice(_TASK_BUILDERS)(rng)

        conversation: Conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": reply})
        yield conversation


def seed_instructions_record(count: int) -> SourceRecord:
    """Provenance for the generated instruction set, labelled as synthetic."""
    return SourceRecord(
        name="bravien-seed-instructions",
        url="local://bravien.data.instructions.generate_seed_conversations",
        license="CC0 1.0 (generated locally by this repository)",
        documents=count,
        processing=["template generation"],
        notes="Synthetic instruction data over the seed corpus lexicon. Teaches "
        "reply format and Bravien's own identity; does not make the model a "
        "useful assistant. Point --data at a real instruction file for that.",
    )


# --------------------------------------------------------------------- splitting


@dataclass
class ConversationSplit:
    train: list[Conversation] = field(default_factory=list)
    validation: list[Conversation] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{len(self.train):,} train / {len(self.validation):,} validation "
            "conversations"
        )


def split_conversations(
    conversations: Iterable[Conversation],
    *,
    val_fraction: float = 0.05,
    min_val: int = 8,
    seed: int = 0,
) -> ConversationSplit:
    """Shuffle and hold out a validation slice.

    A random split is right here, unlike the tail split used for pretraining:
    conversations are independent examples, so neighbouring ones do not leak into
    each other the way adjacent blocks of a concatenated token stream do.
    """
    items = list(conversations)
    if not items:
        raise InstructionDataError("no conversations to split")

    rng = random.Random(seed)
    rng.shuffle(items)

    val_count = max(int(len(items) * val_fraction), min_val)
    if val_count >= len(items):
        # Too few examples to hold any back. Training on all of them and reporting
        # no validation loss is honest; a validation set that is most of the data
        # is not.
        logger.warning(
            "only %d conversation(s): training on all of them with no validation "
            "split, so eval loss will not be reported",
            len(items),
        )
        return ConversationSplit(train=items, validation=[])

    return ConversationSplit(train=items[val_count:], validation=items[:val_count])
