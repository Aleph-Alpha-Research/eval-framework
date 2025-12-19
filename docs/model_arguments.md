# Model Arguments

The Eval-Framework provides a set of model wrapper classes that standardize how LLMs are loaded, formatted, and used for evaluation. Each wrapper manages specific model backends, such as Hugging Face, OpenAI, Aleph Alpha API, or vLLM-based models.

The following sections describe the **constructor arguments** for each model class, highlighting configuration options, defaults, and their purpose. Understanding these arguments allows you to customize evaluation behavior, token limits, concurrency, and model-specific settings.

## HFLLM Class — Constructor Arguments

`HFLLM` is a high-level wrapper for Hugging Face causal language models within the evaluation framework.
It extends `BaseHFLLM`, managing model loading (from local checkpoints, HF Hub, or W&B), formatting, and text generation.

### HFLLM Constructor Argument Reference

| **Argument** | **Type** | **Description** | **Default** |
|:-------------|-----------|-----------------|:------------:|
| `checkpoint_path` | `str \| Path \| None` | Path to a **local checkpoint directory or model weights**. Used when loading from disk instead of HF Hub. | `None` |
| `model_name` | `str \| None` | Hugging Face model name (e.g. `"EleutherAI/pythia-410m"`). Used to load from the Hub. | `None` |
| `artifact_name` | `str \| None` | Weights & Biases artifact name (e.g. `"org/model:latest"`). Used to fetch models from W&B registry. | `None` |
| `formatter` | `BaseFormatter \| None` | Explicit formatter instance used to convert chat messages into model prompts. If not provided, falls back to `DEFAULT_FORMATTER`. |                  `None`                 |
| `formatter_name` | `str \| None` | Name of a formatter class (e.g. `"ConcatFormatter"`, `"HFFormatter"`). Used when `formatter` is not provided. | `None` |
| `formatter_kwargs` | `dict[str, Any] \| None` | Keyword arguments for the formatter constructor (used with `formatter_name`). | `None` |
| `checkpoint_name` | `str \| None` | Custom display/logging name for the checkpoint. If omitted, inferred from model or artifact name. | `None` |
| `bytes_per_token` | `float \| None` | Used to scale token generation limits based on model tokenizer density. Passed to the parent class `BaseHFLLM`. See [Deep Dive: bytes_per_token](#deep-dive-bytes-per-token). | `None` *(internally defaults to `4.0`)* |
| `**kwargs` | `Any` | Additional keyword args passed to `BaseHFLLM` / `BaseLLM`. | — |

---

## AlephAlphaAPIModel — Constructor Arguments

`AlephAlphaAPIModel` is a wrapper around the Aleph Alpha API, extending `BaseLLM`.
It handles formatter setup, request concurrency, retry behavior, and timeout management.

### AlephAlphaAPIModel Constructor Argument Reference

| **Argument**                    | **Type**                | **Description**                                                                                                                   |               **Default**               |
| :------------------------------ | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------: |
| `formatter`                     | `BaseFormatter \| None` | Explicit formatter instance used to convert chat messages into model prompts. If not provided, falls back to `DEFAULT_FORMATTER`. |                  `None`                 |
| `checkpoint_name`               | `str \| None`           | Custom display or logging name for the model. If omitted, uses the class-level `LLM_NAME`.                                        |                  `None`                 |
| `max_retries`                   | `int`                   | Maximum number of retry attempts for failed API requests (e.g. network errors, rate limits).                                      |                  `100`                  |
| `max_async_concurrent_requests` | `int`                   | Maximum number of concurrent asynchronous API requests allowed. Controls throughput and parallelism.                              |                   `32`                  |
| `request_timeout_seconds`       | `int`                   | Maximum number of seconds before an API request times out.                                                                        |    `1805` *(30 minutes + 5 seconds)*    |
| `queue_full_timeout_seconds`    | `int`                   | Maximum number of seconds to wait when the async request queue is full before giving up.                                          |    `1805` *(30 minutes + 5 seconds)*    |
| `bytes_per_token`               | `float \| None`         | Used to scale token-based limits based on model tokenizer density. See [Deep Dive: bytes_per_token](#deep-dive-bytes-per-token).  | `None` *(internally defaults to `4.0`)* |

---

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

## BaseVLLMModel — Constructor Arguments

`BaseVLLMModel` defines the core initialization logic for all vLLM-backed models.
It manages GPU allocation, tokenizer setup, and internal sampling parameter normalization.

### BaseVLLMModel Constructor Argument Reference

| **Argument**             | **Type**                                   | **Description**                                                                                                                             |          **Default**         |
| :----------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------: |
| `formatter`                     | `BaseFormatter \| None` | Explicit formatter instance used to convert chat messages into model prompts. If not provided, falls back to `DEFAULT_FORMATTER`. |                  `None`                 |
| `max_model_len`          | `int \| None`                              | Maximum sequence length (token context limit). Used to configure the vLLM engine.                                                           |            `None`            |
| `tensor_parallel_size`   | `int`                                      | Number of GPUs for tensor-level parallel inference.                                                                                         |              `1`             |
| `gpu_memory_utilization` | `float`                                    | Fraction of total GPU memory reserved for model weights and KV cache.                                                                       |             `0.9`            |
| `batch_size`             | `int`                                      | Number of sequences processed per batch.                                                                                                    |              `1`             |
| `checkpoint_path`        | `str \| Path \| None`                      | Local model path or checkpoint directory.                                                                                                   |            `None`            |
| `checkpoint_name`        | `str \| None`                              | Human-readable identifier for the checkpoint.                                                                                               |            `None`            |
| `sampling_params`        | `SamplingParams \| dict[str, Any] \| None` | Sampling configuration parameters.                                                                                                          |            `None`            |
| `bytes_per_token` | `float \| None`         | Used to scale token-based limits based on model tokenizer density. See [Deep Dive: bytes_per_token](#deep-dive-bytes-per-token).             | `None` *(internally defaults to `4.0`)* |
| `**kwargs`               | `Any`                                      | Any remaining parameters forwarded to the `LLM` engine constructor.                                                                         |               —              |

---

## MistralVLLM — Constructor Arguments

`MistralVLLM` is a specialized subclass of `VLLMModel` → `BaseVLLMModel` designed to run **Mistral** Hugging Face models using the **vLLM** inference backend.
It provides flexible model loading (from local files, Hugging Face Hub, or Weights & Biases), GPU-efficient parallelism, and tunable sampling behavior.

### MistralVLLM Constructor Argument Reference

| **Argument**             | **Type**                                   | **Description**                                                                                                                                                                          |               **Default**               |
| :----------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------: |
| `model_name`             | `str \| None`                              | Hugging Face model identifier (e.g. `"mistralai/Mistral-7B-v0.1"`). Used if `checkpoint_path` is not provided.                                                                           |                  `None`                 |
| `artifact_name`          | `str \| None`                              | **Weights & Biases artifact** reference (e.g. `"org/model:latest"`). Used for pulling models from W&B registry.                                                                          |                  `None`                 |
| `formatter_name`         | `str \| None`                              | Name of a registered formatter class (e.g. `"ConcatFormatter"`, `"HFFormatter"`). Used when `formatter` is not supplied.                                                                 |                  `None`                 |
| `formatter_kwargs`       | `dict[str, Any] \| None`                   | Keyword arguments passed to the formatter constructor.                                                                                                                                   |                  `None`                 |
| `**kwargs`               | `Any`                                      | Additional keyword arguments passed through to `BaseVLLMModel` / `BaseLLM`.                                                                                                              |                    —                    |

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
  model = HFLLM(model_name="my-model", bytes_per_token=3.2)
  ```
* Common approximate values:

  * GPT-family BPE: **3–4 bytes/token**
  * SentencePiece or WordPiece (smaller vocab): **2–3 bytes/token**
  * Character-level tokenizers: **1–2 bytes/token**
