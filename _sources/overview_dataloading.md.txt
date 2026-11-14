## Overview Dataloading

To evaluate models on benchmarks, we compose tasks from reusable parts (`ComposedBenchmark.compose`): a
dataset policy loads the data, an eval kind renders each dataset item into a prompt, and an answer policy
defines how the model's response is obtained. The framework supports two main evaluation types:
**completion tasks** (text generation) and **loglikelihood tasks** (multiple choice ranking).

### From Dataset Item to Scored Response

Each evaluated sample passes through this pipeline:

1. **Loading** — the benchmark's `DatasetPolicy` loads the dataset for one subject at a pinned revision,
   and the rows of the sample split are shuffled with a seeded RNG.
2. **Sample creation** — the benchmark's `EvalKind` turns one dataset item (a `dict` row) into the scored
   sample: the prompt messages, the candidate completions (loglikelihood tasks), the ground truth, and any
   per-sample metric context. The result is a `Sample` (`eval_framework.contract`).
3. **Model response** — the model returns a `RawCompletion` (generated text) or `RawLoglikelihood`
   (a log-probability per candidate), defined in `eval_framework.shared.types`.
4. **Answer extraction** — the `AnswerPolicy` post-processes the response into the scored `Completion` or
   `Loglikelihood`, which the metrics consume.

### Message Structure

Each prompt is structured as a sequence of messages using the template formatting system:

```python
from template_formatting.formatter import Message, Role


class Message(BaseModel):
    role: Role  # SYSTEM, USER, or ASSISTANT
    content: str  # Message content
```

A model-specific formatter renders this message list into the final prompt string (or chat request).

### Prompt Assembly

Eval kinds assemble the message list with `assemble_messages` (`eval_framework.eval_kind`), which builds
the standard prompt shape:

1. **System turn** (optional) — only when the sample carries a system prompt.
2. **Few-shot demonstrations** — the benchmark's `FewShotPolicy` draws examples from the few-shot split
   and renders each as a USER / ASSISTANT pair.
3. **The item's USER turn** — the question or instruction, rendered from the dataset item.
4. **Answer cue** (optional) — a final ASSISTANT turn (e.g. `"Answer:"`) that the model's generation
   continues, used to anchor the expected answer format.

### Example: Geography Quiz

Here's how a complete geography quiz prompt might look (two-shot):

```python
messages = [
    Message(Role.SYSTEM, "Answer geography questions accurately."),
    Message(Role.USER, "Question: What is the capital of Germany?"),
    Message(Role.ASSISTANT, "Answer: Berlin"),
    Message(Role.USER, "Question: What is the capital of France?"),
    Message(Role.ASSISTANT, "Answer: Paris"),
    Message(Role.USER, "Question: What is the capital of Italy?"),
    Message(Role.ASSISTANT, "Answer:"),
]
```

For **completion tasks**, the model generates the complete answer: `" Rome"`

For **loglikelihood tasks**, the model ranks options: `[" Rome", " Madrid", " Athens", " Vienna"]`
