# How to Evaluate HuggingFace Models with Eval Framework

This guide shows you how to evaluate any HuggingFace model using the eval-framework, from simple setup to advanced configurations.

## Quick Start

Here's a sample of code to evaluate a HuggingFace model:

```python
from functools import partial
from pathlib import Path

from eval_framework.llm.huggingface import HFLLM
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig
from template_formatting.formatter import HFFormatter

# Define your model
class MyHuggingFaceModel(HFLLM):
    LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    DEFAULT_FORMATTER = partial(HFFormatter, "meta-llama/Llama-3.2-3B-Instruct")

if __name__ == "__main__":
    # Initialize your model
    llm = MyHuggingFaceModel()

    # Configure evaluation
    config = EvalConfig(
        task_name="ARC",
        num_fewshot=3,
        num_samples=100,
        output_dir=Path("./eval_results"),
        llm_class=MyHuggingFaceModel,
    )

    # Run evaluation
    results = main(llm=llm, config=config)
```

## Understanding the Components

### 1. Model Definition

The `HFLLM` base class provides the foundation for HuggingFace model integration:

```python
class MyModel(HFLLM):
    LLM_NAME = "model-name-on-huggingface"
    DEFAULT_FORMATTER = partial(HFFormatter, "model-name-on-huggingface")

    def __init__(self, formatter=None):
        # Set custom attributes before calling super().__init__
        super().__init__(formatter=formatter)

        # Additional model configuration can be done here
        # Note: model and tokenizer are already loaded in super().__init__
```

### 2. Formatter Selection

The formatter determines how prompts are structured for your model. Choose based on your model type:


#### **Concat Formatter (Base Models):**
```python
from template_formatting.formatter import ConcatFormatter

class BaseModel(HFLLM):
    LLM_NAME = "meta-llama/Llama-3.2-3B"
    DEFAULT_FORMATTER = ConcatFormatter
```
*Simple concatenation formatter for base models without chat templates.*

#### **Llama3 Formatter:**
```python
from template_formatting.formatter import Llama3Formatter

class Llama3Model(HFLLM):
    LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DEFAULT_FORMATTER = Llama3Formatter
```
*Specialized formatter for Llama 3 models with their specific chat template.*

#### **Mistral Formatter:**
```python
from template_formatting.mistral_formatter import MistralFormatter

class MistralModel(HFLLM):
    LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
    DEFAULT_FORMATTER = MistralFormatter
```

#### **Automatic HF Formatter:**
```python
from template_formatting.formatter import HFFormatter
from functools import partial

class ChatModel(HFLLM):
    LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    DEFAULT_FORMATTER = partial(HFFormatter, "meta-llama/Llama-3.2-3B-Instruct")
```
*Automatically detects and uses the model's chat template from HuggingFace.*

## Step-by-Step Implementation

### Step 1: Choose Your Model

Pick any HuggingFace model. Here are examples for different model types:

#### **Large Language Models:**
```python
class Llama3_8B(HFLLM):
    LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DEFAULT_FORMATTER = Llama3Formatter

class Mistral7B(HFLLM):
    LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
    DEFAULT_FORMATTER = MistralFormatter

class Qwen2_7B(HFLLM):
    LLM_NAME = "Qwen/Qwen2-7B-Instruct"
    DEFAULT_FORMATTER = partial(HFFormatter, "Qwen/Qwen2-7B-Instruct")
```

#### **Small Models:**
```python
class SmolLM(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    DEFAULT_FORMATTER = partial(HFFormatter, "HuggingFaceTB/SmolLM-1.7B-Instruct")

class TinyLlama(HFLLM):
    LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_FORMATTER = partial(HFFormatter, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```


### Step 2: Set Up Evaluation Configuration

Configure the evaluation parameters:

```python
from pathlib import Path
from eval_framework.tasks.eval_config import EvalConfig

config = EvalConfig(
    # Core settings
    task_name="MMLU",                    # Benchmark to run
    num_fewshot=5,                       # Number of examples in prompt
    num_samples=100,                     # How many questions to evaluate
    output_dir=Path("./results"),        # Where to save results
    llm_class=YourModelClass,            # Your model class

    # Optional settings
    subjects=["mathematics"],             # Specific subjects (if applicable)
    batch_size=8,                        # Batch processing size
    random_seed=42,                      # Reproducibility
)
```

### Step 3: Run Evaluation

Execute the evaluation:

```python
from eval_framework.main import main

if __name__ == "__main__":
    # Initialize model
    llm = YourModelClass()

    # Run evaluation
    results = main(llm=llm, config=config)

    # Results are automatically saved to output_dir
    print(f"Evaluation completed! Results saved to {config.output_dir}")
```
