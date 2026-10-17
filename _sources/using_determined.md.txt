# Using Determined


> **Note:** These instructions assume you already have access to a running Determined cluster and the necessary permissions to create projects and run experiments. For cluster setup and administration, refer to the [Determined documentation](https://docs.determined.ai/latest/).


[Determined](https://www.determined.ai/) is an open-source platform for distributed and parallelized machine learning experiments. Follow these steps to set up and run evaluations with Determined:

## 1. Install Determined CLI

Install the Determined CLI in your Python environment:

```bash
pip install determined
```

Check the [Determined documentation](https://docs.determined.ai/latest/) for compatible versions with your cluster.

## 2. Connect to Your Determined Cluster

Set the `DET_MASTER` environment variable to the address of your Determined master node:

```bash
export DET_MASTER=<your-determined-master-address>
```

## 3. Login to Determined

Log in with your username:

```bash
det user login <your-username>
```

## 4. Create a Project

Create a new project using the Determined web interface or CLI. Update your experiment configuration files with your project name.

## 5. Prepare Your Experiment Configuration

Arguments to your script are passed as hyperparameters in the experiment config. Example configuration:

```yaml
environment:
  image: "<your-docker-image>"
hyperparameters:
  experiment_name: <EXPERIMENT_NAME>
  task_args:
    type: categorical
    ...
  output_dir: <OUTPUT_DIR>
  llm_name: <LLM_NAME>
  model_path: <MODEL_PATH>
entrypoint: uv run eval_framework --context determined --models <YOUR_MODEL_DEFINITIONS>.py
```

- `<YOUR_MODEL_DEFINITIONS>.py` should define your model configuration.
- See `examples/local_evaluation/` for sample model definition files.

## 6. Set Up Authentication Tokens

Add required tokens as environment variables in your `.env` file:

```bash
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_key_here  # Only needed for LLM-as-a-judge tasks
```

See `.env.example` for a template.

## 7. Run Your Experiment

Start your experiment with:

```bash
det e path/to/your/determined_config.yaml .
```

---

**For more details and advanced usage, see the [Determined documentation](https://docs.determined.ai/latest/)
