# Contextual Awareness in Sequence Processing: An Encoder-Decoder Study using Attention-based Transformers

This repository contains a simple PyTorch implementation of an Encoder-Decoder Transformer architecture designed to demonstrate contextual awareness in sequence processing through attention mechanisms. 

The project uses a synthetic **Sequence Reversal** task to make the learning process fast, intuitive, and easy to visualize. By analyzing the attention weights during decoding, we can clearly see how the model learns to "focus" on specific parts of the input sequence to generate the correct output.

## Project Structure

- `dataset.py`: Contains the `StringReversalDataset` class which generates synthetic integer sequences and their reversals. It also provides a PyTorch `DataLoader` for batching.
- `model.py`: Implements the `ContextFormer` model. This includes custom Positional Encoding, a standard Transformer Encoder, and a custom Transformer Decoder that explicitly returns the cross-attention weights for visualization.
- `train.py`: The main training loop. It trains the model using CrossEntropyLoss and the Adam optimizer.
- `evaluate.py`: Evaluates the trained model by passing an input sequence, predicting the reversed sequence autoregressively, and generating a visualization plot (`attention.png`) of the Encoder-Decoder attention weights.

## Requirements

To run this project, you need Python and PyTorch installed.

```bash
pip install torch numpy matplotlib
```

## Setup and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/meghanasingareddy/ContextFormer-Contextual-Awareness-in-Sequence-Processing-using-Attention-based-Transformers.git
   cd ContextFormer-Contextual-Awareness-in-Sequence-Processing-using-Attention-based-Transformers
   ```

2. **Train the Model**:
   Run the training script to train the model from scratch. The script will automatically generate synthetic data.
   ```bash
   python train.py
   ```
   After training completes, the model weights will be saved to `context_former.pt`.

3. **Evaluate and Visualize Attention**:
   Once the model is trained, run the evaluation script to see it in action.
   ```bash
   python evaluate.py
   ```
   This script will print the input sequence, expected output, and the model's predicted output. It will also generate an `attention.png` file in the same directory. Open this image to visualize the attention map!

## Understanding the Attention Visualization

When you open `attention.png`, you'll see a matrix:
- The **X-axis** represents the input (source) sequence.
- The **Y-axis** represents the generated (target) sequence.
- The color intensity represents the attention weight (how much focus the decoder placed on a specific input token when generating a specific output token).

Since the task is sequence reversal, a successfully trained model should display a strong diagonal line going from the bottom-left to the top-right, showing that to generate the *first* output token, the model looks at the *last* input token, and so on. This is the essence of contextual awareness in sequence processing!
