# Task naming convention

This document describes the naming convention for tasks in the `eval-framework`.
We use the task's **class name** as its identifier, which is what needs to be passed to the
`--task-name` flag, what is logged in results, and what names the output directory.

## Canonical grammar

All class names of tasks should follow the following grammar:

```text
{Dataset}_{Source}_{Language}[_{Style}][_{Variant}][_{Subset}]
```

Fields in square brackets are optional, but we recommend to use the Style field for likelihood tasks (to avoid having to rename them later, when we add a new style).
Note, internal base classes do not need to follow this grammar, but should start with `_` and end with `Base`.

| Field        | Rule                                                                   | Examples                                                          |
| ------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Dataset**  | Shortest sensible, canonical capitalization, no hyphens                | `ARC`, `MMLU`, `HellaSwag`, `GoldenSwag`, `MMLUPro`, `GlobalMMLU` |
| **Source**   | E.g. HF org, capitalized first letter then canonical capitalization    | `AllenAI`, `Ellamind`, `LeoLM`, `EU20`                            |
| **Language** | Prompt language, Two-letter (ISO 639), uppercase, multi-lingual = `XX` | `EN`, `DE`, `FI`                                                  |
| **Style**    | Optional, (e.g. `MC` / `Cloze` / `BPB` / `PartialEval`)                | `Cloze`                                                           |
| **Variant**  | Optional, after Style (e.g. `OLMES`, `IDK`, `COT`)                     | `OLMES`                                                           |
| **Subset**   | Optional trailing subset                                               | `Diamond`                                                         |

**Dataset + Source + Language are mandatory** and always present, in that order.

Examples:

| Canonical class name       | Meaning                                   |
| -------------------------- | ----------------------------------------- |
| `ARC_AllenAI_EN_Cloze`     | ARC, AllenAI source, English, Cloze style |
| `ARC_AllenAI_EN_MC_OLMES`  | …multiple-choice, OLMES variant           |
| `ARC_AllenAI_EN_Cloze_IDK` | …Cloze, "I don't know" variant            |
| `ARC_LeoLM_DE_Cloze`       | German ARC, LeoLM translation, Cloze      |
| `ARC_Ellamind_DE_Cloze`    | German ARC, Ellamind translation, Cloze   |

## Naming notes

Additional notes on the naming convention:

- **MC vs MC_OLMES.** If a dataset has a single multiple-choice task, name it `_MC`. Only when both a default MC and an OLMES-style MC (with space before each label in the prompt) exist do we distinguish `_MC` and `_MC_OLMES`.
- **Source is usually the HF org**, canonical-capitalized (`allenai`→`AllenAI`). However, sometimes dataset live in a personal HF account which might not be the institution we associate with that particular dataset. In these cases, use the originating **institution** when there is a single clear one and the HF path is just a mirror/personal handle (e.g. `SQuAD` → `Stanford`, not `rajpurkar`); when a dataset's variants come from different repos, name each by **its own** source (e.g. DROP completion → `EleutherAI`, MC/Cloze → `AllenAI`). If we do use the person, and the handle is a clear name, use capitalization for the first letter of the first and surname, e.g. `mandarjoshi` → `MandarJoshi`.
- **COT.** A chain-of-thought multiple-choice task is `_COTMC` (single token): it answers with a letter but is generation-scored, not a likelihood-MC.
- **Multilingual.** A single class spanning many languages uses `XX`; per-language subclasses use the language code (`GlobalMMLU_Cohere_XX_MC` → `GlobalMMLU_Cohere_DE_MC`).
