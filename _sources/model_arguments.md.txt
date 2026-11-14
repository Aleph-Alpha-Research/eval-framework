# Model Arguments

The Eval-Framework provides a set of model wrapper classes that standardize how LLMs are loaded, formatted, and used for evaluation. Each wrapper manages specific model backends, such as OpenAI-compatible endpoints.

The following sections describe the **constructor arguments** for each model class, highlighting configuration options, defaults, and their purpose. Understanding these arguments allows you to customize evaluation behavior, token limits, concurrency, and model-specific settings.

## OpenAIModel — Constructor Arguments

`OpenAIModel` is a wrapper for OpenAI’s API models (e.g., GPT-4, GPT-3.5) that integrates with the evaluation framework.
It manages model configuration, authentication, and request parameters for the OpenAI client.

### OpenAIModel Constructor Argument Reference

| **Argument**      | **Type**                | **Description**                                                                                                                              |               **Default**               |
| :---------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------: |
| `model_name`      | `str`                   | Name of the **OpenAI model** to use (e.g. `"gpt-4o"`, `"gpt-4"`, `"gpt-3.5-turbo"`). Determines which model endpoint to call.                |                `"gpt-4o"`               |
| `formatter`                     | `BaseFormatter \| None` | Explicit formatter instance used to convert chat messages into model prompts. If not provided, falls back to `DEFAULT_FORMATTER`. |                  `None`                 |
| `temperature`     | `float \| None`         | Sampling temperature controlling output randomness (`0.0–2.0`). Lower = deterministic, higher = creative.                                    |     `None` *(interpreted as `0.0`)*     |
| `api_key`         | `str \| None`           | OpenAI API key. If not provided, defaults to the `OPENAI_API_KEY` environment variable.                                                      |                  `None`                 |
| `organization`    | `str \| None`           | Optional OpenAI **organization ID** for multi-org API usage or billing separation.                                                           |                  `None`                 |
| `base_url`        | `str \| None`           | Custom **API base URL**, e.g., for Azure OpenAI endpoints or local proxies.                                                                  |                  `None`                 |
| `bytes_per_token` | `float \| None`         | Used to scale token-based limits based on model tokenizer density. See [Deep Dive: bytes_per_token](#deep-dive-bytes-per-token).             | `None` *(internally defaults to `4.0`)* |

---

## Deep Dive: `bytes_per_token`

### What it is
`bytes_per_token` is a scalar that adjusts generation limits (`max_tokens`) based on the model’s tokenizer characteristics.
Different models tokenize text differently — some produce more tokens per byte, some fewer.
This parameter helps keep **generation length consistent** across models by normalizing the token budget.

### How it works internally
```python
if bytes_per_token is not None:
    bytes_per_token_scalar = 4.0 / bytes_per_token
else:
    bytes_per_token_scalar = 4.0 / BYTES_PER_TOKEN  # defaults to 4.0 / 4.0 = 1.0
````

* The constant `BYTES_PER_TOKEN = 4.0` is a heuristic from OpenAI’s tokenizer documentation.
* The scalar is then applied when calculating generation limits:

  ```python
  scaled_max_tokens = ceil(max_tokens * bytes_per_token_scalar)
  ```
* This ensures that, for models with different token byte sizes, output lengths remain roughly comparable in bytes or visible characters.

### Why it matters

Without this scaling, a model that uses *shorter tokens* would produce *longer outputs* (more tokens for the same byte size),
and a model with *longer tokens* would produce *shorter outputs*.
`bytes_per_token` compensates for that, aligning models to a common byte-level generation length.

### Example calculations

| **Scenario**                      | **bytes_per_token** | **Computed Scalar (4.0 / bpt)** | **max_tokens=100 → scaled** | **Effect**                               |
| --------------------------------- | ------------------- | ------------------------------- | --------------------------- | ---------------------------------------- |
| Default (no override)             | `None` → 4.0        | `1.0`                           | `100`                       | No change.                               |
| Denser tokenizer (2 bytes/token)  | `2.0`               | `2.0`                           | `200`                       | Model can generate twice as many tokens. |
| Sparser tokenizer (8 bytes/token) | `8.0`               | `0.5`                           | `50`                        | Model generates half as many tokens.     |

### Recommended usage

* Leave unset (`None`) for most models — default behavior is fine.
* If you empirically know your tokenizer’s average bytes/token, pass it explicitly:

  ```python
  model = OpenAIModel(model_name="my-model", bytes_per_token=3.2)
  ```
* Common approximate values:

  * GPT-family BPE: **3–4 bytes/token**
  * SentencePiece or WordPiece (smaller vocab): **2–3 bytes/token**
  * Character-level tokenizers: **1–2 bytes/token**
