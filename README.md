# HyGende

HyGende is a framework for **Hy**pothesis **Gen**eration for **DE**pression detection using Large Language Models (LLMs). It automates the process of discovering rules or hypotheses from labeled training data and applying them to make predictions on new data.

## Features

- **Hypothesis Generation**: iteratively generates and refines hypotheses based on training examples.
- **Inference Strategies**: Supports multiple inference methods using the generated hypotheses:
  - `WeightBasedInference`: Weighted voting based on hypothesis ranking.
  - `RelevanceBasedInference`: Dynamically selects relevant hypotheses for each case.
  - `RandomInference`: Random selection.
- **LLM Support**: Flexible backend support for various LLMs including:
  - vLLM (local deployment)
  - OpenAI GPT series
  - Anthropic Claude
  - HuggingFace models
- **Task Configurable**: Easily extensible to new text classification tasks via YAML configuration.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd hygende
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `hygende/`: Core library code.
  - `algorithm/`: Algorithms for generation, inference, and updating hypotheses.
  - `LLM_wrapper/`: Wrappers for different LLM APIs.
  - `tasks.py`: Task management and data loading.
  - `prompt.py`: Prompt construction and template management.
- `hygende_cmd/`: Detailed running scripts.
  - `generation.py`: Script to run the hypothesis generation process.
  - `inference.py`: Script to run inference using generated hypotheses.
- `examples/`: Quick example for hypothesis generation and inference.
- `data/`: Dataset storage and task configurations (e.g., `config.yaml`).

## Usage

### 1. Configuration
Prepare your data and a `config.yaml` file. See `data/paddle/config.yaml` for an example. The config defines data paths and prompt templates.

### 2. Hypothesis Generation
Use the generation script to create a bank of hypotheses from your training data.

```bash
python -m hygende_cmd.generation \
    --task_config_path ./data/paddle/config.yaml \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_folder ./outputs/paddle \
    --num_train 200 \
    --max_num_hypotheses 20
```

### 3. Inference
Once you have a generated hypothesis file (JSON), use it to run inference on the test set.

```bash
python -m hygende_cmd.inference \
    --task_config_path ./data/paddle/config.yaml \
    --hypothesis_file ./outputs/paddle/hypotheses_final.json \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --inference_type weight
```

