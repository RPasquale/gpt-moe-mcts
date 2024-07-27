# GPT-MoE-MCTS: GPT with Mixture of Experts and Monte Carlo Tree Search

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [MCTS Decoding](#mcts-decoding)
9. [Contributing](#contributing)


## Introduction

GPT-MoE-MCTS is an advanced language model that combines the power of GPT (Generative Pre-trained Transformer) with Mixture of Experts (MoE) and Monte Carlo Tree Search (MCTS) decoding. This model is designed to provide high-quality text generation with improved efficiency and performance.

## Key Features

- **GPT-based Architecture**: Utilizes the powerful GPT architecture for language modeling.
- **Mixture of Experts**: Incorporates a dynamic routing system to specialize different parts of the network for different inputs.
- **FlashAttention3**: Implements an optimized attention mechanism for improved efficiency.
- **Monte Carlo Tree Search Decoding**: Uses MCTS during inference for higher quality text generation.
- **Hugging Face Compatible**: Easily integrates with the Hugging Face Transformers library.

## Model Architecture

The GPT-MoE-MCTS model consists of the following key components:

1. **Token and Positional Embeddings**: Converts input tokens into embeddings and adds positional information.
2. **Transformer Blocks with MoE**: Multiple layers of transformer blocks, each incorporating:
   - FlashAttention3: An optimized attention mechanism.
   - Mixture of Experts Layer: A dynamic routing system for specialized processing.
   - Feed-Forward Network: Standard MLP for additional processing.
3. **Output Layer**: Final layer normalization and projection to vocabulary logits.

## Installation

To install the GPT-MoE-MCTS model, follow these steps:

```bash
git clone https://github.com/RPasquale/gpt-moe-mcts.git
cd gpt-moe-mcts
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the GPT-MoE-MCTS model:

```python
from transformers import GPT2Tokenizer
from modeling_gpt_moe_mcts import GPTMoEMCTSModel
from configuration_gpt_moe_mcts import GPTMoEMCTSConfig

# Initialize configuration and model
config = GPTMoEMCTSConfig()
model = GPTMoEMCTSModel(config)

# Initialize tokenizer (using GPT2Tokenizer as a base)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Prepare input
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
outputs = model(**inputs)

# Get the predicted next token
next_token_logits = outputs.logits[0, -1, :]
next_token = next_token_logits.argmax()

# Decode the predicted token
predicted_text = tokenizer.decode(next_token)

print(f"Input: {text}")
print(f"Predicted next token: {predicted_text}")
```

## Training

To train the GPT-MoE-MCTS model on your own data:

1. Prepare your dataset in the format of tokenized .npy files.
2. Adjust the hyperparameters in the `train_model()` function in `train.py`.
3. Run the training script:

```bash
python train.py
```

The script will automatically save checkpoints and display training progress.

## Evaluation

To evaluate the model's performance:

```python
from eval_utils import evaluate_model

perplexity, accuracy = evaluate_model(model, eval_dataloader)
print(f"Perplexity: {perplexity}, Accuracy: {accuracy}")
```

## MCTS Decoding

The GPT-MoE-MCTS model uses Monte Carlo Tree Search for decoding during inference. To use MCTS decoding:

```python
from mcts_decode import mcts_decode

generated_text = mcts_decode(model, input_text, max_length=50, num_simulations=100)
print(f"Generated text: {generated_text}")
```

## Contributing

We welcome contributions to the GPT-MoE-MCTS project! If you're interested in contributing, please visit our [GitHub repository](https://github.com/RPasquale/gpt-moe-mcts) for more information on how to get involved. You can submit issues, feature requests, or pull requests there.


---

For more detailed information about the model architecture, training process, and advanced usage, please refer to our [documentation](docs/index.md).

If you use GPT-MoE-MCTS in your research, please cite:

```
@misc{GPT-MoE-MCTS,
  author = {Robbie Pasquale},
  title = {GPT-MoE-MCTS: GPT with Mixture of Experts and Monte Carlo Tree Search},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RPasquale/gpt-moe-mcts}},
  version = {1.0.0},
  note = {This project is currently in development.}
}
```