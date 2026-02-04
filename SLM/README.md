# GPT-2 Inference

Refactored GPT-2 model inference code with clean organization.

## Structure

```
.
├── models/
│   ├── __init__.py
│   ├── attention.py      # Multi-head attention
│   ├── layers.py         # LayerNorm, GELU, FeedForward
│   └── gpt.py           # TransformerBlock, GPTModel
├── utils/
│   ├── __init__.py
│   ├── dataset.py       # Dataset utilities
│   ├── tokenizer.py     # Tokenizer utilities
│   └── generation.py    # Text generation
├── config.py            # Model configurations
├── inference.py         # Main inference script
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Inference

```python
from inference import load_model, run_inference
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, config = load_model("path/to/model.pth", device=device)

response = run_inference(
    model, 
    config, 
    instruction="What is the formula for speed?",
    device=device
)
print(response)
```

### Run Examples

```bash
python inference.py
```

Update `MODEL_PATH` in `inference.py` to point to your model file.

## Model Configurations

Supported models:
- gpt2-small (124M)
- gpt2-medium (355M)
- gpt2-large (774M)
- gpt2-xl (1558M)

## Dataset

Trained on: [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
