# 🧠 ContextFormer

**Contextual Awareness in Sequence Processing**  
*An Encoder-Decoder Study using Attention-based Transformers*

---

## Overview

ContextFormer is an educational deep-learning project that demonstrates how an **Encoder-Decoder Transformer** uses cross-attention to learn the **reverse ordering** of integer sequences. The Streamlit application provides a live, interactive dashboard suitable for academic lab presentations.

This is a **synthetic sequence-reversal experiment**, not a general-purpose language model. The model predicts sequences in reverse order and the cross-attention heatmap visualizes how each decoder position attends to encoder positions.

---

## Architecture Overview

```
Input Sequence → Token Embedding + Positional Encoding
              → Encoder Self-Attention (×2 layers)
              → Decoder Masked Self-Attention (×2 layers)
              → Decoder Cross-Attention (×2 layers)   ← visualized
              → Output Projection (Linear → Vocab)
              → Predicted Sequence (greedy decoding)
```

| Parameter       | Value  |
|-----------------|--------|
| Vocabulary size | 30     |
| d_model         | 64     |
| Attention heads | 4      |
| Encoder layers  | 2      |
| Decoder layers  | 2      |
| FFN hidden dim  | 256    |
| Max seq length  | 15     |

Special tokens: `PAD=0`, `SOS=1`, `EOS=2`. Data tokens: integers `3–29`.

---

## Project Structure

```
ContextFormer .../
├── model.py            # ContextFormer architecture (Encoder-Decoder Transformer)
├── dataset.py          # Synthetic reversal dataset generator
├── train.py            # Training loop (Adam, CrossEntropy, 15 epochs)
├── evaluate.py         # Evaluation + static attention plot (attention.png)
├── app.py              # Streamlit presentation application
├── context_former.pt   # Trained model checkpoint
├── attention.png       # Static attention heatmap (from evaluate.py)
└── requirements.txt    # Python dependencies
```

---

## Environment Setup

> **Windows PowerShell note:** The project directory name contains spaces.
> Always quote the path or `cd` into it before running commands.

### 1. Create and activate a virtual environment

```powershell
# Navigate to the project (note the quoted path)
cd "c:\meghna\Coding\proj\ContextFormer Contextual Awareness in Sequence Processing using Attention-based Transformers"

# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## Training

```powershell
python train.py
```

Training runs for 15 epochs (≈ 1–2 minutes on CPU). The checkpoint is saved to `context_former.pt` in the project directory.

Expected output:
```
Using device: cpu
Starting training...
Epoch: 01 | Train Loss: 2.340 | Val Loss: 1.987
...
Epoch: 15 | Train Loss: 0.210 | Val Loss: 0.235
Model saved to context_former.pt
```

---

## Running the Streamlit Application

```powershell
python -m streamlit run app.py --server.port 8502
```

Open your browser at **http://localhost:8502**.

The application features four tabs:
1. **Sequence Experiment** — Input, expected output, model prediction, token-level accuracy.
2. **Attention Visualization** — Live cross-attention heatmap with per-position analysis.
3. **Transformer Architecture** — Detailed explanation of the model implementation.
4. **About the Experiment** — Dataset, training, evaluation, and limitations.

---

## Evaluation (Static)

```powershell
python evaluate.py
```

Runs inference on a fixed example sequence and saves `attention.png`.

---

## Attention Heatmap

The cross-attention heatmap shows how each decoder output position attends to encoder input positions:

- **X-axis:** Source tokens (encoder input)
- **Y-axis:** Target tokens (decoder output, including `⟨SOS⟩`)
- **Bright cell:** Strong attention weight at that (target, source) pair
- **Ideal pattern:** An inverted diagonal — output position `i` attends to input position `N-1-i`

Real learned attention maps rarely form a perfect diagonal because attention distributes weight softly across all positions, and positional encodings give the model indirect position information.

> ⚠️ Attention weights are not a complete explanation of model reasoning. They show which inputs were weighted, not which inputs caused the output.

---

## Known Limitations

- The model uses **greedy decoding** (no beam search). Some predictions will be incorrect.
- Training for 15 epochs on 10,000 samples produces a model with partial accuracy.
- The vocabulary is tiny (30 tokens) and the task is purely synthetic.
- Attention is displayed as the **average across all heads**. Individual heads may differ.
- Training loss and validation loss are **not stored** in the checkpoint. Run `train.py` with additional logging if you need curves.
- The model displays its real outputs — mismatches are intentional and informative.

---

## License

Educational project. Not intended for production use.
