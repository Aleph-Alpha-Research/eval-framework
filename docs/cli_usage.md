# Using CLI

The Eval-Framework CLI provides a flexible interface for evaluating LLMs across a wide range of benchmarks. Whether you're running evaluations locally or in a distributed environment, the CLI allows you to configure tasks, models, and metrics with ease.


```
usage: eval_framework [-h] [--context {local,determined}] --models MODELS [--llm-name LLM_NAME] [--llm-args [LLM_ARGS ...]] [--num-samples NUM_SAMPLES] [--num-fewshot NUM_FEWSHOT] [--task-name TASK_NAME] [--task-subjects TASK_SUBJECTS [TASK_SUBJECTS ...]] [--judge-models JUDGE_MODELS] [--output-dir OUTPUT_DIR] [--batch-size BATCH_SIZE] [--llm-judge-name LLM_JUDGE_NAME]

options:
  -h, --help
        show this help message and exit

  --context {local,determined}
        The context in which the evaluation is run.

  --models MODELS
        The path to the Python module file containing model classes.

  --llm-name LLM_NAME
        The class derived from `eval_framework.llm.base.BaseLLM` found in the models module to instantiate for evaluation.

  --llm-args [LLM_ARGS ...]
        The arguments to pass to the LLM as key=value pairs.

  --num-samples NUM_SAMPLES
        The number of samples per subject to evaluate.

  --num-fewshot NUM_FEWSHOT
        The number of fewshot examples to use.

  --max-tokens
        The maximum number of tokens to generate for each sample. Overwrites any task default value.

  --task-name TASK_NAME
        The name of the task to evaluate.

  --hf-revision HF_REVISION
        A tag name, a branch name, or commit hash for the task HF dataset.

  --wandb-project WANDB_PROJECT
      The name of the Weights & Biases project to log runs to.

  --wandb-entity WANDB_ENTITY
      The name of the Weights & Biases entity to log runs to.

  --wandb-run-id WANDB_RUN_ID
      The ID of an existing Weights & Biases run to resume.

  --task-subjects TASK_SUBJECTS [TASK_SUBJECTS ...]
        The subjects of the task to evaluate. If empty, all subjects are evaluated. Subjects in the form of tuples can be specified in a comma-delimited way, possibly using wildcard * in some dimensions of a tuple. Examples: "DE_DE, *" "FR_FR, astronomy".

  --judge-models JUDGE_MODELS
        The path to the Python module file containing LLM judge model classes.

  --output-dir OUTPUT_DIR
        The path for the evaluation outputs.

  --batch-size BATCH_SIZE
        Size of batch of samples to send to the LLM for evaluation in parallel.Use 1 for sequential running (default) and None to have a single batch.

  --judge-model-name JUDGE_MODEL_NAME
        The class derived from `eval_framework.llm.base.BaseLLM` found in the judge-models module to instantiate for LLM judge evaluation metrics.

  --judge-model-args JUDGE_MODEL_ARGS
        The args of the judge model used within OpenAIModel wrapper.

  --perturbation-type TYPE
        The type of to apply to task intstructions. Note that this may not make sense especially for prompts containing math and code.

  --perturbation-probability PROBABILITY
        The probability of applying perturbation to each word or character (between 0.0 and 1.0).

  --perturbation-seed SEED
        Random seed controlling perturbations.

  --description DESCRIPTION
        Description of the run. This will be added to the metadata of the run to help with bookkeeping.
```

Execute a single evaluation locally by running:

```bash
poetry run eval_framework \
    --models src/eval_framework/llm/models.py \
    --llm-name Llama31_8B_Instruct_HF \
    --task-name "GSM8K" \
    --output-dir ./eval \
    --num-fewshot 5 \
    --num-samples 10
```

## Running Hugging Face Models

You can also run models directly from Hugging Face Hub using the `HFLLM_from_name` class:

```bash
poetry run eval_framework \
    --models src/eval_framework/llm/models.py \
    --llm-name HFLLM_from_name \
    --llm-args model_name="microsoft/DialoGPT-medium" formatter="Llama3Formatter" \
    --task-name "GSM8K" \
    --output-dir ./eval \
    --num-fewshot 5 \
    --num-samples 10
```

This approach allows you to evaluate any model available on Hugging Face by specifying the `model_name` and appropriate `formatter` in the `--llm-args` parameter.
