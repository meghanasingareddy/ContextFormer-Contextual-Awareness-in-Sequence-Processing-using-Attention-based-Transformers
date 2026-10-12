"""
ContextFormer — Presentation-Ready Streamlit Application
Encoder-Decoder Transformer for Sequence Reversal with Cross-Attention Visualization
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextFormer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark academic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0e1117;
    color: #dde3f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13192b;
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #a8b8d8;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Metric cards ── */
[data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    word-break: break-all;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #7a8fb0 !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Cards ── */
.cf-card {
    background: #151d2e;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.cf-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5b7ab0;
    margin-bottom: 8px;
}
.cf-card-seq {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 1.0rem;
    color: #e8edf8;
    word-break: break-word;
    line-height: 1.7;
}
.cf-badge-match {
    background: #0d3b27;
    color: #3de87a;
    border: 1px solid #1f6644;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
}
.cf-badge-mismatch {
    background: #3b1a0d;
    color: #f0803a;
    border: 1px solid #7a3010;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
}
.cf-token-match   { color: #3de87a; font-weight: bold; }
.cf-token-miss    { color: #f0803a; font-weight: bold;
                    text-decoration: underline wavy #f0803a; }
.cf-token-neutral { color: #8ba5cc; }

/* ── Section headings ── */
.cf-section {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6fa5;
    border-left: 3px solid #2d4f8a;
    padding-left: 10px;
    margin: 18px 0 10px 0;
}

/* ── Architecture flow ── */
.cf-arch {
    background: #111826;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 0.85rem;
    color: #6e9ecf;
    font-family: monospace;
    text-align: center;
    letter-spacing: 0.05em;
}

/* ── Metric row ── */
.cf-metric-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin: 10px 0;
}
.cf-metric-pill {
    background: #151d2e;
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #8ba5cc;
}
.cf-metric-pill strong { color: #c8d8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must match training config exactly)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE        = 30
D_MODEL           = 64
NHEAD             = 4
NUM_ENC_LAYERS    = 2
NUM_DEC_LAYERS    = 2
DIM_FEEDFORWARD   = 256
MAX_SEQ_LEN       = 15   # max content token length (not counting SOS/EOS)
PAD_IDX           = 0
SOS_IDX           = 1
EOS_IDX           = 2
VALID_TOKEN_MIN   = 3
VALID_TOKEN_MAX   = VOCAB_SIZE - 1   # inclusive

CHECKPOINT_PATH   = os.path.join(os.path.dirname(__file__), "context_former.pt")

PRESETS = [
    [10, 15, 20, 25, 5, 8],
    [7, 14, 21, 28, 9],
    [12, 4, 19, 23, 11, 16, 6],
    [3, 8, 15, 22, 29],
    [5, 10, 3, 27, 18, 9, 14, 21],
]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from model import ContextFormer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextFormer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=100,
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        return model, device, False, "Checkpoint file not found. Run `python train.py` first."

    try:
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, device, True, None
    except Exception as e:
        return model, device, False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, src_tokens: list, device) -> tuple:
    """
    Returns:
        pred_seq       : list[int]  – predicted content tokens (no SOS/EOS)
        tgt_all        : list[int]  – full decoding including SOS and optional EOS
        layer_attns    : list[np.ndarray] shape (tgt_len, src_len) per decoder layer
    """
    model.eval()
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
    tgt_indices = [SOS_IDX]

    with torch.no_grad():
        for _ in range(MAX_SEQ_LEN + 4):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
            output, attns = model(src_tensor, tgt_tensor)
            tok = int(output.argmax(2)[:, -1].item())
            tgt_indices.append(tok)
            if tok == EOS_IDX:
                break

    # Extract and convert attention tensors → numpy
    # attns: list[Tensor] each shape (1, tgt_len, src_len)
    layer_attns = [a.squeeze(0).cpu().numpy() for a in attns]

    has_eos = tgt_indices[-1] == EOS_IDX
    pred_seq = tgt_indices[1:-1] if has_eos else tgt_indices[1:]
    return pred_seq, tgt_indices, layer_attns


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_seq(tokens: list) -> str:
    return "[" + ", ".join(str(int(t)) for t in tokens) + "]"


def token_accuracy(pred: list, expected: list) -> tuple[int, int]:
    n = max(len(pred), len(expected))
    correct = sum(1 for i in range(n)
                  if i < len(pred) and i < len(expected) and int(pred[i]) == int(expected[i]))
    return correct, n


def render_comparison_html(pred: list, expected: list) -> str:
    pred_i    = [int(x) for x in pred]
    expected_i = [int(x) for x in expected]
    n = max(len(pred_i), len(expected_i))
    tokens_html = []
    for i in range(n):
        p = pred_i[i] if i < len(pred_i) else None
        e = expected_i[i] if i < len(expected_i) else None
        if p is None:
            tokens_html.append(f'<span class="cf-token-miss">?</span>')
        elif e is None:
            tokens_html.append(f'<span class="cf-token-miss">{p}</span>')
        elif p == e:
            tokens_html.append(f'<span class="cf-token-match">{p}</span>')
        else:
            tokens_html.append(f'<span class="cf-token-miss">{p}</span>')
    return "[ " + ", ".join(tokens_html) + " ]"


def make_heatmap(attn_matrix: np.ndarray, src_labels: list, tgt_labels: list,
                 layer_idx: int, cmap: str) -> plt.Figure:
    valid_rows = min(len(tgt_labels), attn_matrix.shape[0])
    valid_cols = min(len(src_labels), attn_matrix.shape[1])
    sub = attn_matrix[:valid_rows, :valid_cols]

    fig_w = max(5, valid_cols * 0.7 + 2)
    fig_h = max(4, valid_rows * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(sub, cmap=cmap, aspect="auto", vmin=0, vmax=sub.max() or 1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color="#6e9ecf")
    cbar.outline.set_edgecolor("#1e2d4a")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8ba5cc", fontsize=8)
    cbar.set_label("Attention weight", color="#8ba5cc", fontsize=9)

    ax.set_xticks(np.arange(valid_cols))
    ax.set_yticks(np.arange(valid_rows))
    ax.set_xticklabels(src_labels[:valid_cols], fontsize=10, color="#c8d8f0",
                       fontfamily="monospace")
    ax.set_yticklabels(tgt_labels[:valid_rows], fontsize=10, color="#c8d8f0",
                       fontfamily="monospace")
    ax.tick_params(colors="#4a6fa5", length=3)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Add numeric annotations if small enough
    if valid_rows <= 12 and valid_cols <= 12:
        for r in range(valid_rows):
            for c in range(valid_cols):
                v = sub[r, c]
                color = "white" if v < 0.5 * sub.max() else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color, fontfamily="monospace")

    ax.set_xlabel("Source tokens  (Encoder input)", color="#6e9ecf", fontsize=10, labelpad=6)
    ax.set_ylabel("Target tokens  (Decoder output)", color="#6e9ecf", fontsize=10, labelpad=8)
    ax.set_title(f"Decoder Layer {layer_idx + 1} — Cross-Attention Weights",
                 color="#8ba5cc", fontsize=11, pad=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d4a")

    fig.tight_layout(pad=0.8)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #1e2d4a; padding-bottom:16px; margin-bottom:20px;">
  <h1 style="margin:0; font-size:2rem; color:#c8d8f0; font-weight:800; letter-spacing:0.01em;">
    🧠 ContextFormer
  </h1>
  <p style="margin:4px 0 2px 0; font-size:1.05rem; color:#6e9ecf; font-weight:500;">
    Contextual Awareness in Sequence Processing
  </p>
  <p style="margin:0; font-size:0.82rem; color:#4a6fa5; font-style:italic; letter-spacing:0.04em;">
    An Encoder-Decoder Study using Attention-based Transformers
  </p>
</div>
""", unsafe_allow_html=True)

# Architecture overview & Linguistic Context Motivation
st.markdown("""
<div class="cf-arch">
  Input Sequence &nbsp;→&nbsp; <span style="color:#3a7bd5;">Encoder Self-Attention</span>
  &nbsp;→&nbsp; <span style="color:#8a4fff;">Decoder Cross-Attention</span>
  &nbsp;→&nbsp; <span style="color:#3de87a;">Predicted Sequence</span>
</div>
<p style="font-size:0.8rem; color:#4a6fa5; margin:8px 0 12px 0;">
  This project investigates <strong>Contextual Awareness</strong> in sequence processing. While large language models use contextual attention to disambiguate words and resolve pronouns, ContextFormer isolates and tests this foundational mechanism on an algorithmic sequence task (sequence reversal).
</p>
""", unsafe_allow_html=True)

with st.expander("💡 Conceptual Intuition: Why Context Matters (Presentation & PPT Guide)", expanded=False):
    st.markdown("""
### 🗣️ Context in Natural Language vs. Sequence Processing
In human language and transformer NLP architectures, words have no fixed meaning in isolation—their representation is dynamically shaped by surrounding tokens through **self-attention**:

| # | Sentence Example | Contextual Attention Mechanism |
|---|-------------------|--------------------------------|
| **1** | *"The bank was near the river."* | **Polysemy Disambiguation:** Context ("river") clarifies that *"bank"* refers to the riverbank, not a financial institution. |
| **2** | *"The animal didn't cross the street because it was tired."* | **Coreference Resolution:** The model attends from *"it"* to *"animal"* (as animals experience fatigue). |
| **3** | *"The boy ate an apple because he was hungry."* | **Pronoun Binding:** *"He"* dynamically links back to *"boy"* through subject-predicate context. |
| **4** | *"She sat by the window because the room was hot."* | **Causal Context:** *"The room"* provides the contextual rationale for selecting the window. |
| **5** | *"The dog chased the cat because it was running."* | **Ambiguity Resolution:** Attention weights can flexibly distribute between *"dog"* or *"cat"* based on syntactic structure. |
| **6** | *"The teacher praised the student because she worked hard."* | **Coreference Resolution:** Syntactic context attributes *"she"* to the *"student"*. |
| **7** | *"The glass fell off the table and it broke."* | **Anaphora Resolution:** *"It"* attaches to *"glass"* based on physical affordances. |
| **8** | *"The man put the umbrella away because the rain stopped."* | **Situational Context:** The causal clause explains the action on *"umbrella"*. |

---
#### 🎯 Recommended Slide / Demo Highlights for your Presentation:
1. **Word Sense Disambiguation:**  
   > *“The bank was near the river.”*  
   *Demonstrates how attention transforms ambiguous lexical representations into context-specific embeddings.*
2. **Pronoun & Anaphora Binding (Winograd Schema):**  
   > *“The animal didn't cross the street because it was tired.”*  
   *Demonstrates how contextual attention identifies antecedent entities across sentence boundaries.*
3. **ContextFormer's Role in this Study:**  
   Instead of black-box language modeling, **ContextFormer** formalizes this exact contextual routing mechanism into an **Encoder-Decoder Transformer** reversing sequences—making cross-attention weights directly measurable and transparent on a heatmap!
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading ContextFormer checkpoint…"):
    model, device, is_loaded, load_err = load_model()

if not is_loaded:
    st.error(f"**Checkpoint error:** {load_err}")
    st.stop()

device_str = "CUDA (GPU)" if str(device) == "cuda" else "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Input Panel
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-size:0.72rem; font-weight:800; letter-spacing:0.1em;
            text-transform:uppercase; color:#4a6fa5; margin-bottom:12px;">
  ⚙️ Input Settings
</div>
""", unsafe_allow_html=True)

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Preset Example", "Custom Sequence", "Random Generator"],
    label_visibility="collapsed",
)

src_tokens = None
input_error = None

if input_mode == "Preset Example":
    st.sidebar.markdown(
        "<div style='font-size:0.75rem;color:#5b7ab0;margin-bottom:4px;'>Choose a preset</div>",
        unsafe_allow_html=True,
    )
    choice_idx = st.sidebar.selectbox(
        "Preset",
        range(len(PRESETS)),
        format_func=lambda i: fmt_seq(PRESETS[i]),
        label_visibility="collapsed",
    )
    src_tokens = PRESETS[choice_idx]

elif input_mode == "Custom Sequence":
    st.sidebar.markdown(
        f"<div style='font-size:0.75rem;color:#5b7ab0;margin-bottom:4px;'>"
        f"Integers {VALID_TOKEN_MIN}–{VALID_TOKEN_MAX}, length 1–{MAX_SEQ_LEN}</div>",
        unsafe_allow_html=True,
    )
    raw = st.sidebar.text_input(
        "Custom",
        value="10, 15, 20, 25, 5, 8",
        placeholder="e.g. 10, 15, 20, 25",
        label_visibility="collapsed",
    )
    try:
        parts = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
        if not parts:
            input_error = "Please enter at least one integer."
        else:
            tokens = [int(p) for p in parts]
            bad = [t for t in tokens if not (VALID_TOKEN_MIN <= t <= VALID_TOKEN_MAX)]
            if bad:
                input_error = (
                    f"Token(s) {bad} are outside the valid range "
                    f"[{VALID_TOKEN_MIN}, {VALID_TOKEN_MAX}]. "
                    f"PAD=0, SOS=1, EOS=2 are reserved."
                )
            elif len(tokens) > MAX_SEQ_LEN:
                input_error = f"Sequence too long ({len(tokens)}). Max is {MAX_SEQ_LEN}."
            else:
                src_tokens = tokens
    except ValueError:
        input_error = "Non-integer value detected. Use comma-separated integers only."

else:  # Random Generator
    seq_len = st.sidebar.slider(
        "Length", min_value=2, max_value=MAX_SEQ_LEN, value=6
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)
    if st.sidebar.button("🎲 Generate"):
        rng = np.random.default_rng(int(seed))
        st.session_state.random_seq = rng.integers(
            VALID_TOKEN_MIN, VALID_TOKEN_MAX + 1, size=seq_len
        ).tolist()
    if "random_seq" not in st.session_state:
        rng = np.random.default_rng(42)
        st.session_state.random_seq = rng.integers(
            VALID_TOKEN_MIN, VALID_TOKEN_MAX + 1, size=seq_len
        ).tolist()
    src_tokens = [int(x) for x in st.session_state.random_seq]

st.sidebar.divider()

# ── Heatmap controls ──
st.sidebar.markdown(
    "<div style='font-size:0.72rem;font-weight:800;letter-spacing:0.1em;"
    "text-transform:uppercase;color:#4a6fa5;margin-bottom:8px;'>🎨 Heatmap</div>",
    unsafe_allow_html=True,
)
layer_idx = st.sidebar.selectbox(
    "Decoder layer",
    options=list(range(NUM_DEC_LAYERS)),
    format_func=lambda i: f"Layer {i + 1}",
)
colormap = st.sidebar.selectbox(
    "Colormap",
    ["viridis", "plasma", "magma", "Blues", "coolwarm", "bone", "YlOrRd"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_exp, tab_attn, tab_arch, tab_about = st.tabs([
    "🔬 Sequence Experiment",
    "🔍 Attention Visualization",
    "🏗️ Transformer Architecture",
    "📄 About the Experiment",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Sequence Experiment
# ══════════════════════════════════════════════════════════════════
with tab_exp:
    if input_error:
        st.error(f"**Input Error:** {input_error}")
        st.stop()

    if src_tokens is None:
        st.info("Configure input in the sidebar.")
        st.stop()

    # Run inference
    with st.spinner("Running ContextFormer inference…"):
        pred_seq, tgt_all, layer_attns = run_inference(model, src_tokens, device)

    expected_seq   = src_tokens[::-1]
    pred_seq_int   = [int(x) for x in pred_seq]
    expected_int   = [int(x) for x in expected_seq]
    exact_match    = pred_seq_int == expected_int
    correct_toks, total_toks = token_accuracy(pred_seq_int, expected_int)

    # ── Sequence cards ──
    st.markdown('<div class="cf-section">Sequence Results</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="cf-card">
          <div class="cf-card-title">📥 Input Sequence</div>
          <div class="cf-card-seq">{fmt_seq(src_tokens)}</div>
          <div style="font-size:0.73rem;color:#4a6fa5;margin-top:6px;">
            Length: {len(src_tokens)} tokens
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="cf-card">
          <div class="cf-card-title">🎯 Expected Target</div>
          <div class="cf-card-seq">{fmt_seq(expected_seq)}</div>
          <div style="font-size:0.73rem;color:#4a6fa5;margin-top:6px;">
            Perfect reversal of input
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        badge = (
            '<span class="cf-badge-match">✅ Exact Match</span>'
            if exact_match
            else '<span class="cf-badge-mismatch">⚠️ Partial Match</span>'
        )
        comparison_html = render_comparison_html(pred_seq_int, expected_int)
        st.markdown(f"""
        <div class="cf-card">
          <div class="cf-card-title">🤖 Model Prediction &nbsp;{badge}</div>
          <div class="cf-card-seq">{comparison_html}</div>
          <div style="font-size:0.73rem;color:#4a6fa5;margin-top:6px;">
            Length: {len(pred_seq_int)} tokens
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Accuracy metrics ──
    st.markdown('<div class="cf-section">Performance Metrics</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Device", device_str)
    mc2.metric("Exact Match", "Yes ✅" if exact_match else "No ⚠️")
    mc3.metric("Token Accuracy", f"{correct_toks}/{total_toks}")
    mc4.metric("Correct Tokens", f"{correct_toks}")
    mc5.metric("Mismatched", f"{total_toks - correct_toks}")

    if not exact_match:
        mismatches = []
        n = max(len(pred_seq_int), len(expected_int))
        for i in range(n):
            p = pred_seq_int[i] if i < len(pred_seq_int) else "—"
            e = expected_int[i] if i < len(expected_int) else "—"
            if p != e:
                mismatches.append(f"pos {i}: got **{p}**, expected **{e}**")
        if mismatches:
            with st.expander("🔎 Mismatch Details"):
                for m in mismatches:
                    st.markdown(f"- {m}")
                st.caption(
                    "Mismatches are normal — this is a live model output, not a hardcoded answer. "
                    "Train for more epochs to improve accuracy."
                )
    else:
        st.success("The model correctly reversed the entire sequence! 🎉")

    st.caption(
        f"Training metrics (loss curves) are not stored in the checkpoint. "
        f"Run `python train.py` with logging to capture them."
    )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Attention Visualization
# ══════════════════════════════════════════════════════════════════
with tab_attn:
    if input_error or src_tokens is None:
        st.info("Configure a valid input in the sidebar to see attention.")
    else:
        if "pred_seq_int" not in dir():
            with st.spinner("Running inference for attention…"):
                pred_seq, tgt_all, layer_attns = run_inference(model, src_tokens, device)
                pred_seq_int   = [int(x) for x in pred_seq]
                expected_int   = [int(x) for x in src_tokens[::-1]]
                exact_match    = pred_seq_int == expected_int

        st.markdown('<div class="cf-section">Cross-Attention Heatmap</div>', unsafe_allow_html=True)

        attn_matrix = layer_attns[layer_idx]  # (tgt_len, src_len)

        # Build labels
        tgt_labels = ["⟨SOS⟩"] + [str(int(x)) for x in pred_seq_int]
        has_eos = tgt_all[-1] == EOS_IDX
        if has_eos:
            tgt_labels.append("⟨EOS⟩")
        src_labels = [str(int(x)) for x in src_tokens]

        col_plot, col_info = st.columns([1.5, 1], gap="large")

        with col_plot:
            fig = make_heatmap(attn_matrix, src_labels, tgt_labels, layer_idx, colormap)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Token-by-token attention summary
            valid_rows = min(len(tgt_labels), attn_matrix.shape[0])
            valid_cols = min(len(src_labels), attn_matrix.shape[1])
            sub = attn_matrix[:valid_rows, :valid_cols]

            with st.expander("📊 Per-Position Peak Attention"):
                for r in range(valid_rows):
                    peak_col = int(np.argmax(sub[r]))
                    peak_val = float(sub[r, peak_col])
                    src_lbl = src_labels[peak_col] if peak_col < len(src_labels) else "?"
                    tgt_lbl = tgt_labels[r]
                    st.markdown(
                        f"**{tgt_lbl}** → attends most to src[{peak_col}] = `{src_lbl}` "
                        f"*(weight {peak_val:.3f})*"
                    )

        with col_info:
            st.markdown("### 💡 Reading the Heatmap")
            st.markdown(f"""
**X-axis** — Source tokens fed into the Encoder.  
**Y-axis** — Target tokens generated by the Decoder (including ⟨SOS⟩).

**Colour intensity** — Cross-attention weight (0 → 1).  
A brighter cell means the decoder position strongly attended to that encoder position.

**Decoder layer shown:** Layer {layer_idx + 1} of {NUM_DEC_LAYERS}

---

**Reversal task pattern:**  
For perfect reversal, an idealized map traces an **inverted diagonal** —  
the first output token should attend to the *last* input token, and so on.

Real learned attention maps rarely form a perfect diagonal because:
- Attention aggregates information across multiple heads.
- The model distributes weight softly, not as hard one-hot assignments.
- Positional encodings give the model indirect position information.

---

> ⚠️ Attention weights are **not** a complete explanation of model reasoning.
> They show which inputs were *weighted*, not which inputs *caused* the output.
""")

            st.markdown("### 📐 Tensor Dimensions")
            st.markdown(f"""
| Dimension | Value |
|-----------|-------|
| Attention shape | `{attn_matrix.shape}` |
| Source length | `{min(len(src_labels), attn_matrix.shape[1])}` |
| Target length | `{min(len(tgt_labels), attn_matrix.shape[0])}` |
| Heads | {NHEAD} (averaged) |
| Decoder layers | {NUM_DEC_LAYERS} |
""")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Architecture
# ══════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### 🏗️ ContextFormer Architecture")
    st.markdown("""
The architecture implemented in `model.py` is a standard **Encoder-Decoder Transformer**,
adapted to keep decoder cross-attention weights accessible for visualization.
""")

    with st.expander("📦 Embedding + Positional Encoding", expanded=True):
        st.markdown(f"""
- **Token Embedding:** `nn.Embedding(vocab_size={VOCAB_SIZE}, d_model={D_MODEL})`  
  Maps integer tokens to dense vectors. Embeddings are scaled by √d_model = {math.sqrt(D_MODEL):.2f}.
- **Positional Encoding:** Sinusoidal fixed encoding added to embeddings.  
  Uses `sin` for even dimensions and `cos` for odd dimensions:
""")
        st.markdown(r"""
$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
  PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Both encoder and decoder share the same positional encoder.
""")

    with st.expander("🔷 Encoder (Self-Attention)"):
        st.markdown(f"""
- **Layers:** {NUM_ENC_LAYERS} × `TransformerEncoderLayer`
- **Multi-head self-attention:** {NHEAD} heads, d_model={D_MODEL}
- **Feed-forward:** Linear({D_MODEL} → {DIM_FEEDFORWARD} → {D_MODEL}) with ReLU
- **Layer Norm + Residual** connections after each sub-layer
- Produces a **memory** tensor of shape `(src_len, batch, {D_MODEL})`
""")

    with st.expander("🔶 Decoder (Masked Self-Attn + Cross-Attn)"):
        st.markdown(f"""
- **Layers:** {NUM_DEC_LAYERS} × `TransformerDecoderLayerWithAttention` (custom class)
- **Sub-layer 1 — Masked Self-Attention:**  
  Prevents each decoder position from attending to future positions via a causal mask.
- **Sub-layer 2 — Cross-Attention:**  
  The decoder queries the encoder memory.  
  Keys and Values come from the encoder output; Queries come from the decoder.  
  Returns `attn_weights` of shape `(batch, tgt_len, src_len)` (heads averaged).
- **Sub-layer 3 — Feed-Forward:** Linear({D_MODEL} → {DIM_FEEDFORWARD} → {D_MODEL})
- **Layer Norm + Residual** after each sub-layer
""")

    with st.expander("🎯 Output Projection + Autoregressive Decoding"):
        st.markdown(f"""
- **Output projection:** `nn.Linear({D_MODEL}, {VOCAB_SIZE})`  
  Projects decoder output to a logit distribution over the vocabulary.
- **Greedy decoding:** At each step, the token with the highest logit is selected.  
  This continues until `EOS` (token {EOS_IDX}) is produced or max length is reached.
- **Special tokens:**

| Token | Index |
|-------|-------|
| PAD   | {PAD_IDX} |
| SOS   | {SOS_IDX} |
| EOS   | {EOS_IDX} |
| Data  | 3 – {VOCAB_SIZE - 1} |
""")

    with st.expander("🔢 Model Configuration"):
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Vocabulary size | {VOCAB_SIZE} |
| d_model | {D_MODEL} |
| Attention heads | {NHEAD} |
| Encoder layers | {NUM_ENC_LAYERS} |
| Decoder layers | {NUM_DEC_LAYERS} |
| FFN hidden dim | {DIM_FEEDFORWARD} |
| Max sequence length | {MAX_SEQ_LEN} |
| Checkpoint | `context_former.pt` |
| Device | {device_str} |
""")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### 📄 About ContextFormer")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
#### 🎯 Objective
Demonstrate that an Encoder-Decoder Transformer can learn the **reverse ordering** of
integer sequences using only the attention mechanism, without any explicit sorting logic.

#### 💡 The Intuition of Contextual Awareness
Why do we study contextual awareness? In natural language and transformer models, words are not static tokens—their semantic interpretation is dictated by surrounding context:
1. **Word Sense Disambiguation:** *"The bank was near the river."* (Context: *"bank"* means riverbank, not a financial institution).
2. **Pronoun Reference (Winograd Schema):** *"The animal didn't cross the street because it was tired."* (Context: *"it"* resolves to the animal).
3. **Coreference Resolution:** *"The boy ate an apple because he was hungry."* (Context: *"he"* refers to the boy).
4. **Causal Reasoning:** *"She sat by the window because the room was hot."* (Context: *"the room"* explains the choice of window).
5. **Structural Disambiguation:** *"The dog chased the cat because it was running."* (Context: *"it"* can reference either based on syntactic focus).
6. **Student Praising:** *"The teacher praised the student because she worked hard."* (Context: *"she"* attributes to the student).
7. **Physical Anaphora:** *"The glass fell off the table and it broke."* (Context: *"it"* refers to the glass).
8. **Action Justification:** *"The man put the umbrella away because the rain stopped."* (Context: explains putting away the umbrella).

**ContextFormer** translates this foundational concept of dynamically weighting context into an algorithmic sequence reversal task where attention weights are explicitly observable on heatmaps!

#### 📂 Synthetic Dataset
- Sequences of random integers sampled from `[3, 29]` (avoiding special token indices).
- Lengths vary uniformly in `[1, MAX_SEQ_LEN)`.
- 10,000 training samples, 80 / 20 train-validation split.
- Target sequences prepend `SOS (1)` and append `EOS (2)`.
- Shorter sequences are padded with `PAD (0)` to a fixed tensor length.

#### 🏋️ Training
- Optimizer: Adam (lr = 0.005)
- Loss: Cross-Entropy, ignoring `PAD` tokens
- Gradient clipping: 1.0
- Epochs: 15
- Batch size: 128
""")

    with col_b:
        st.markdown("""
#### 📊 Evaluation
- **Exact-sequence accuracy:** fraction of test sequences where every token matches.
- **Token-level accuracy:** fraction of individual tokens correctly predicted.
- This interface shows **real model outputs** — mismatches are expected and intentional.

#### ⚠️ Limitations
- The model uses **greedy decoding** (no beam search), which can miss optimal predictions.
- Training for only 15 epochs on a small dataset means some sequences will be wrong.
- The vocabulary is tiny (30 tokens) and the task is purely synthetic.
- Attention weights are **averaged** across heads before display — individual heads may
  show different specialisation patterns.
- This is an educational demonstration, not a production system.

#### 📁 Project Files
| File | Purpose |
|------|---------|
| `model.py` | ContextFormer architecture |
| `dataset.py` | Synthetic reversal dataset |
| `train.py` | Training loop |
| `evaluate.py` | Evaluation + static attention plot |
| `app.py` | This Streamlit application |
| `context_former.pt` | Trained checkpoint |
| `attention.png` | Static attention figure |
""")
