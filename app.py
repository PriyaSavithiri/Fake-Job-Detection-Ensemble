import sys

try:
    import tf_keras
    import tf_keras.preprocessing.text     as _tft
    import tf_keras.preprocessing.sequence as _tfs
    sys.modules["keras.preprocessing.text"]              = _tft
    sys.modules["keras.preprocessing.sequence"]          = _tfs
    sys.modules["keras.preprocessing"]                   = tf_keras.preprocessing
    sys.modules["keras"]                                 = tf_keras
    from tf_keras.models import load_model               as _load_model
    from tf_keras.preprocessing.sequence import pad_sequences as _pad_sequences
    _KERAS_BACKEND = "tf_keras"
except Exception as _e1:
    try:
        import tensorflow.keras
        import tensorflow.keras.preprocessing.text     as _tft
        import tensorflow.keras.preprocessing.sequence as _tfs
        sys.modules["keras.preprocessing.text"]              = _tft
        sys.modules["keras.preprocessing.sequence"]          = _tfs
        sys.modules["keras.preprocessing"]                   = tensorflow.keras.preprocessing
        sys.modules["keras"]                                 = tensorflow.keras
        from tensorflow.keras.models import load_model       as _load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad_sequences
        _KERAS_BACKEND = "tensorflow.keras"
    except Exception as _e2:
        _load_model    = None
        _pad_sequences = None
        _KERAS_BACKEND = f"FAILED: {_e2}"

import streamlit as st
import numpy as np
import pandas as pd
import re
import json
import pickle
import joblib
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="FraudLens — Job Posting Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_load_model    = None
_pad_sequences = None
_KERAS_BACKEND = "not loaded"

try:
    import tf_keras
    import tf_keras.preprocessing.text as _tft
    import tf_keras.preprocessing.sequence as _tfs
    import sys
    sys.modules["keras.preprocessing.text"]     = _tft
    sys.modules["keras.preprocessing.sequence"] = _tfs
    from tf_keras.models import load_model as _load_model
    from tf_keras.preprocessing.sequence import pad_sequences as _pad_sequences
    _KERAS_BACKEND = "tf_keras"
except Exception:
    try:
        from tensorflow.keras.models import load_model as _load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad_sequences
        _KERAS_BACKEND = "tensorflow.keras"
    except Exception as _e:
        _KERAS_BACKEND = f"FAILED: {_e}"

_LIME_OK  = False
_LIME_ERR = ""
_LimeTextExplainer = None
try:
    from lime.lime_text import LimeTextExplainer as _LimeTextExplainer
    _LIME_OK = True
except Exception as _e:
    _LIME_ERR = str(_e)

# ── NLP / Transformers ────────────────────────────────────────────────────────
import nltk
for _p in ['punkt_tab', 'punkt', 'stopwords', 'wordnet']:
    nltk.download(_p, quiet=True)
from nltk.corpus import stopwords as _sw
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --fraud-red:#C0392B;--legit-green:#1A7A4A;--accent:#E8A838;
    --bg-dark:#0D0D0D;--bg-card:#161616;--bg-card2:#1E1E1E;
    --border:#2A2A2A;--text-dim:#888;--text-main:#E8E8E0;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg-dark)!important;color:var(--text-main)!important;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;}
section[data-testid="stSidebar"]{background:var(--bg-card)!important;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] *{color:var(--text-main)!important;}
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem;}
.card-accent{border-left:3px solid var(--accent);}
.verdict-fraud{background:linear-gradient(135deg,#2A0A0A,#1A0808);border:1px solid var(--fraud-red);border-radius:16px;padding:2rem;text-align:center;}
.verdict-legit{background:linear-gradient(135deg,#071A0F,#050F08);border:1px solid var(--legit-green);border-radius:16px;padding:2rem;text-align:center;}
.verdict-title{font-family:'DM Serif Display',serif;font-size:3.2rem;margin:0;letter-spacing:-1px;}
.verdict-fraud .verdict-title{color:#E55;}
.verdict-legit .verdict-title{color:#4C9;}
.verdict-sub{font-family:'DM Mono',monospace;font-size:0.85rem;color:var(--text-dim);margin-top:0.4rem;}
.prob-bar-wrap{margin:0.6rem 0;}
.prob-label{font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--text-dim);margin-bottom:3px;}
.prob-bar-bg{background:#1E1E1E;border-radius:999px;height:8px;overflow:hidden;border:1px solid var(--border);}
.prob-bar-fill-fraud{background:linear-gradient(90deg,#8B1A1A,#C0392B);border-radius:999px;height:100%;}
.prob-bar-fill-legit{background:linear-gradient(90deg,#0D5C34,#1A7A4A);border-radius:999px;height:100%;}
.section-title{font-family:'DM Serif Display',serif;font-size:1.4rem;color:var(--text-main);margin:1.5rem 0 0.8rem;border-bottom:1px solid var(--border);padding-bottom:0.4rem;}
textarea{background:var(--bg-card2)!important;border:1px solid var(--border)!important;color:var(--text-main)!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;}
textarea:focus{border-color:var(--accent)!important;}
.stButton>button{background:var(--accent)!important;color:#000!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;border:none!important;border-radius:8px!important;padding:0.6rem 2rem!important;font-size:1rem!important;transition:opacity 0.2s;}
.stButton>button:hover{opacity:0.88!important;}
.word-pos{background:rgba(26,122,74,0.35);border-radius:3px;padding:1px 3px;color:#5dbb8a;}
.word-neg{background:rgba(192,57,43,0.35);border-radius:3px;padding:1px 3px;color:#e07070;}
.word-neu{color:var(--text-dim);}
.metric-row{display:flex;gap:12px;flex-wrap:wrap;margin:0.8rem 0;}
.metric-box{flex:1;min-width:100px;background:var(--bg-card2);border:1px solid var(--border);border-radius:8px;padding:0.8rem 1rem;text-align:center;}
.metric-val{font-family:'DM Serif Display',serif;font-size:1.6rem;color:var(--accent);}
.metric-lbl{font-family:'DM Mono',monospace;font-size:0.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.08em;}
.lime-table{width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;font-size:0.78rem;}
.lime-table th{color:var(--text-dim);font-weight:400;padding:4px 8px;border-bottom:1px solid var(--border);text-align:left;}
.lime-table td{padding:4px 8px;}
.lime-bar-pos{background:rgba(26,122,74,0.5);height:10px;border-radius:2px;}
.lime-bar-neg{background:rgba(192,57,43,0.5);height:10px;border-radius:2px;}
.upload-hint{text-align:center;color:var(--text-dim);font-size:0.8rem;font-family:'DM Mono',monospace;margin-top:0.4rem;}
</style>
""", unsafe_allow_html=True)

MODELS_PATH = Path(r"NLP/Capstone/job-postings-fraud/models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_INFO = {
    "Naive Bayes":         {"desc": "TF-IDF + Multinomial NB",  "speed": "⚡ Fast"},
    "LSTM":                {"desc": "Bidirectional LSTM",        "speed": "⚡ Fast"},
    "MiniLM":              {"desc": "Transformer (33M params)",  "speed": "🔶 Medium"},
    "RoBERTa":             {"desc": "Transformer (125M params)", "speed": "🔴 Slow"},
    "Ensemble (Stacking)": {"desc": "RF meta-learner (all 4)",   "speed": "🔴 Slow"},
}

_lemmatizer = WordNetLemmatizer()
_stop_words = set(_sw.words('english'))

def preprocess_text(text):
    if not text: return ""
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in _stop_words]
    return ' '.join(_lemmatizer.lemmatize(t) for t in tokens)

def clean_text_lstm(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    M = {}

    # NB
    try:
        M["nb_pipeline"]   = joblib.load(MODELS_PATH / "nb_pipeline.pkl")
        M["nb_vectorizer"] = M["nb_pipeline"].named_steps["tfidfvectorizer"]
        M["nb_model"]      = M["nb_pipeline"].named_steps["multinomialnb"]
    except Exception as e:
        M["nb_error"] = str(e)

    try:
        import h5py
        import numpy as np

        with open(MODELS_PATH / "lstm_tokenizer.pkl", "rb") as f:
            M["lstm_tokenizer"] = pickle.load(f)
        with open(MODELS_PATH / "lstm_config.json") as f:
            cfg = json.load(f)
            M["lstm_max_len"] = cfg["MAX_SEQUENCE_LENGTH"]

        # Build model architecture manually in tf_keras
        FAST_MODE     = cfg.get("FAST_MODE", True)
        EMBEDDING_DIM = cfg.get("EMBEDDING_DIM", 64) 
        LSTM_UNITS    = 64 if FAST_MODE else 128
        DENSE_UNITS   = 32 if FAST_MODE else 64

        import tf_keras
        model = tf_keras.Sequential([
            tf_keras.layers.Embedding(input_dim=10000, output_dim=EMBEDDING_DIM,
                                    input_length=M["lstm_max_len"]),
            tf_keras.layers.LSTM(LSTM_UNITS),
            tf_keras.layers.Dropout(0.3),
            tf_keras.layers.Dense(DENSE_UNITS, activation='relu'),
            tf_keras.layers.Dense(1, activation='sigmoid')
        ])
        model.build(input_shape=(None, M["lstm_max_len"]))

        # Load weights directly from h5 layer by layer
        with h5py.File(MODELS_PATH / "lstm_model.h5", "r") as f:
            model_weights = f.get("model_weights") or f

            layer_map = {
                "embedding": model.layers[0],
                "lstm":      model.layers[1],
                "dense":     model.layers[3],
                "dense_1":   model.layers[4],
            }

            for layer_name, layer in layer_map.items():
                # Try both h5 structures
                grp = (model_weights.get(layer_name) or
                    model_weights.get(f"{layer_name}/{layer_name}"))
                if grp is None:
                    continue
                w_names = grp.attrs.get("weight_names", [])
                if len(w_names) == 0:
                    # Try direct dataset children
                    w_names = list(grp.keys())
                weights = [grp[n][()] if isinstance(grp[n], h5py.Dataset)
                        else grp[n][list(grp[n].keys())[0]][()]
                        for n in w_names]
                if weights:
                    layer.set_weights(weights)

        M["lstm_model"] = model
        test = np.zeros((1, M["lstm_max_len"]))
        _ = model.predict(test, verbose=0)
        print("LSTM loaded via h5py ✓")

    except Exception as e:
        import traceback
        M["lstm_error"] = f"h5py load failed: {e}\n{traceback.format_exc()}"

    # MiniLM
    try:
        M["minilm_tok"] = AutoTokenizer.from_pretrained(str(MODELS_PATH / "model_miniLM_final"))
        M["minilm_mdl"] = AutoModelForSequenceClassification.from_pretrained(
            str(MODELS_PATH / "model_miniLM_final")).to(DEVICE)
        M["minilm_mdl"].eval()
    except Exception as e:
        M["minilm_error"] = str(e)

    # RoBERTa
    try:
        M["roberta_tok"] = AutoTokenizer.from_pretrained(str(MODELS_PATH / "model_roberta_final"))
        M["roberta_mdl"] = AutoModelForSequenceClassification.from_pretrained(
            str(MODELS_PATH / "model_roberta_final")).to(DEVICE)
        M["roberta_mdl"].eval()
    except Exception as e:
        M["roberta_error"] = str(e)

    try:
        M["meta_rf"] = joblib.load(MODELS_PATH / "ensemble_meta_rf.pkl")
        M["meta_lr"] = joblib.load(MODELS_PATH / "ensemble_meta_lr.pkl")
        with open(MODELS_PATH / "ensemble_config.json") as f:
            M["ensemble_config"] = json.load(f)
    except Exception as e:
        M["ensemble_error"] = str(e)

    return M

# ── Prediction functions ──────────────────────────────────────────────────────
def predict_nb(text, M):
    if "nb_error" in M: return None
    try:
        return float(M["nb_pipeline"].predict_proba([preprocess_text(text)])[0][1])
    except Exception:
        return None

def predict_lstm(text, M):
    if "lstm_error" in M or _pad_sequences is None: return None
    try:
        seq    = M["lstm_tokenizer"].texts_to_sequences([clean_text_lstm(text)])
        padded = _pad_sequences(seq, maxlen=M["lstm_max_len"], padding='post', truncating='post')
        return float(M["lstm_model"].predict(padded, verbose=0)[0, 0])
    except Exception:
        return None

def predict_minilm(text, M):
    if "minilm_error" in M: return None
    try:
        enc = M["minilm_tok"](text, return_tensors="pt", truncation=True,
                              max_length=256, padding="max_length").to(DEVICE)
        with torch.no_grad():
            p = torch.softmax(M["minilm_mdl"](**enc).logits, dim=1).cpu().numpy()[0]
        return float(p[1])
    except Exception:
        return None

def predict_roberta(text, M):
    if "roberta_error" in M: return None
    try:
        enc = M["roberta_tok"](text, return_tensors="pt", truncation=True,
                               max_length=256, padding="max_length").to(DEVICE)
        with torch.no_grad():
            p = torch.softmax(M["roberta_mdl"](**enc).logits, dim=1).cpu().numpy()[0]
        return float(p[1])
    except Exception:
        return None

def predict_ensemble(text, M):

    if "ensemble_error" in M: return None
    try:
        nb_p   = predict_nb(text, M)      or 0.5
        lstm_p = predict_lstm(text, M)    or 0.5
        mini_p = predict_minilm(text, M)  or 0.5
        rob_p  = predict_roberta(text, M) or 0.5
        X = np.array([[nb_p, lstm_p, mini_p, rob_p]])
        return float(M["meta_rf"].predict_proba(X)[0][1])
    except Exception:
        return None

PREDICT_FNS = {
    "Naive Bayes":         predict_nb,
    "LSTM":                predict_lstm,
    "MiniLM":              predict_minilm,
    "RoBERTa":             predict_roberta,
    "Ensemble (Stacking)": predict_ensemble,
}

# ── LIME (cached) ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def run_lime_cached(text, model_name, num_features=10, num_samples=300):
    # KEY FIX 4: Use module-level _LimeTextExplainer — no import inside fn
    if not _LIME_OK:
        return f"LIME import failed: {_LIME_ERR}"
    try:
        M = load_all_models()

        def _nb_batch(texts):
            return np.vstack([
                M["nb_pipeline"].predict_proba([preprocess_text(t)])[0] for t in texts
            ])

        def _lstm_batch(texts):
            if _pad_sequences is None or "lstm_error" in M:
                n = len(texts)
                return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
            seqs   = M["lstm_tokenizer"].texts_to_sequences([clean_text_lstm(t) for t in texts])
            padded = _pad_sequences(seqs, maxlen=M["lstm_max_len"], padding='post', truncating='post')
            p = M["lstm_model"].predict(padded, verbose=0).flatten()
            return np.column_stack([1 - p, p])

        def _mini_batch(texts):
            enc = M["minilm_tok"](list(texts), return_tensors="pt",
                                  padding=True, truncation=True, max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(M["minilm_mdl"](**enc).logits, dim=1).cpu().numpy()
            return p

        def _rob_batch(texts):
            enc = M["roberta_tok"](list(texts), return_tensors="pt",
                                   padding=True, truncation=True, max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(M["roberta_mdl"](**enc).logits, dim=1).cpu().numpy()
            return p

        def _ens_batch(texts):
            rows = []
            for t in texts:
                rows.append([
                    predict_nb(t, M)      or 0.5,
                    predict_lstm(t, M)    or 0.5,
                    predict_minilm(t, M)  or 0.5,
                    predict_roberta(t, M) or 0.5,
                ])
            return M["meta_rf"].predict_proba(np.array(rows))

        batch_fns = {
            "Naive Bayes":         (_nb_batch,   _LimeTextExplainer(class_names=["legit", "fraud"])),
            "LSTM":                (_lstm_batch,  _LimeTextExplainer(class_names=["legit", "fraud"], bow=False, split_expression=r'\W+', random_state=42)),
            "MiniLM":              (_mini_batch,  _LimeTextExplainer(class_names=["legit", "fraud"], bow=False, split_expression=r'\W+', random_state=42)),
            "RoBERTa":             (_rob_batch,   _LimeTextExplainer(class_names=["legit", "fraud"], bow=False, split_expression=r'\W+', random_state=42)),
            "Ensemble (Stacking)": (_ens_batch,   _LimeTextExplainer(class_names=["legit", "fraud"], bow=False, split_expression=r'\W+', random_state=42)),
        }
        fn, explainer = batch_fns[model_name]
        exp = explainer.explain_instance(
            text, fn, num_features=num_features,
            num_samples=num_samples, labels=(1,)
        )
        return exp.as_list(label=1)

    except Exception as e:
        import traceback
        return f"LIME error: {e}\n{traceback.format_exc()}"

# ── UI helpers ────────────────────────────────────────────────────────────────
def prob_bar(label, prob, is_fraud=True):
    pct   = prob * 100
    cls   = "prob-bar-fill-fraud" if is_fraud else "prob-bar-fill-legit"
    color = '#e07070' if is_fraud else '#5dbb8a'
    st.markdown(f"""
    <div class="prob-bar-wrap">
      <div class="prob-label">{label}&nbsp;
        <strong style="color:{color}">{pct:.1f}%</strong>
      </div>
      <div class="prob-bar-bg">
        <div class="{cls}" style="width:{pct:.1f}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_verdict(prob, model_name):
    is_fraud = prob >= 0.5
    verdict  = "⚠ FRAUD" if is_fraud else "✓ LEGIT"
    cls      = "verdict-fraud" if is_fraud else "verdict-legit"
    conf     = abs(prob - 0.5) * 2 * 100
    conf_lbl = "HIGH" if conf > 70 else "MEDIUM" if conf > 40 else "LOW"
    st.markdown(f"""
    <div class="{cls}">
      <p class="verdict-title">{verdict}</p>
      <p class="verdict-sub">
        fraud probability: {prob*100:.1f}% &nbsp;·&nbsp;
        confidence: {conf_lbl} &nbsp;·&nbsp; model: {model_name}
      </p>
    </div>""", unsafe_allow_html=True)


def render_lime(lime_results):
    if isinstance(lime_results, str):
        st.error(lime_results)
        return
    if not lime_results:
        st.info("No LIME results.")
        return

    sorted_r = sorted(lime_results, key=lambda x: abs(x[1]), reverse=True)

    # Build a proper dataframe — no HTML needed
    rows = []
    for word, val in sorted_r:
        rows.append({
            "Token":     word,
            "Direction": "↑ FRAUD" if val > 0 else "↓ LEGIT",
            "Score":     round(val, 4),
            "Impact":    abs(val),
        })

    df = pd.DataFrame(rows)

    def color_score(val):
        color = "#e07070" if val > 0 else "#5dbb8a"
        return f"color: {color}; font-weight: 500"

    def color_direction(val):
        color = "#e07070" if "FRAUD" in val else "#5dbb8a"
        return f"color: {color}"

    styled = (
        df.style
        .applymap(color_score, subset=["Score"])
        .applymap(color_direction, subset=["Direction"])
        .bar(subset=["Impact"], color=["#5dbb8a", "#e07070"], vmin=0)
        .format({"Score": "{:+.4f}", "Impact": "{:.4f}"})
        .hide(axis="index")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    words_dict = {w: v for w, v in lime_results}
    raw_text   = st.session_state.get("last_text", "")
    tokens     = re.findall(r'\S+|\s+', raw_text)
    highlighted = ""
    for tok in tokens:
        s = tok.strip()
        if s in words_dict:
            v = words_dict[s]
            if v > 0:
                style = "background:rgba(192,57,43,0.35);border-radius:3px;padding:1px 3px;color:#e07070"
            else:
                style = "background:rgba(26,122,74,0.35);border-radius:3px;padding:1px 3px;color:#5dbb8a"
            highlighted += f'<span style="{style}">{tok}</span>'
        else:
            highlighted += f'<span style="color:#888">{tok}</span>'

    with st.expander("📝 Highlighted job text", expanded=False):
        st.markdown(
            f'<div style="font-size:0.85rem;line-height:1.9;font-family:sans-serif">{highlighted}</div>',
            unsafe_allow_html=True
        )


def render_all_models_chart(scores):
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor='#161616')
    ax.set_facecolor('#161616')
    names  = list(scores.keys())
    values = list(scores.values())
    colors = ['#C0392B' if v >= 0.5 else '#1A7A4A' for v in values]
    bars   = ax.barh(names, values, color=colors, height=0.55, edgecolor='none')
    ax.axvline(0.5, color='#E8A838', linewidth=1.2, linestyle='--', alpha=0.7, label='Decision boundary')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraud probability", color='#888', fontsize=9)
    ax.tick_params(colors='#888', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2A2A2A')
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val*100:.1f}%', va='center', color='#E8E8E0', fontsize=8.5)
    ax.legend(fontsize=8, labelcolor='#888', facecolor='#161616', edgecolor='#2A2A2A')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem">
      <p style="font-family:'DM Serif Display',serif;font-size:1.8rem;margin:0;color:#E8E8E0;line-height:1.1">
        Fraud<span style="color:#E8A838">Lens</span></p>
      <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#888;margin:0.3rem 0 0">
        Job Posting Fraud Detector</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.72rem;color:#888;'
        'text-transform:uppercase;letter-spacing:0.1em">Model Selection</p>',
        unsafe_allow_html=True
    )
    use_best = st.toggle("🏆 Use Best Model (Ensemble)", value=True)

    if not use_best:
        selected_model = st.radio("Choose model", list(MODEL_INFO.keys()))
        info = MODEL_INFO[selected_model]
        st.markdown(f"""
        <div class="card" style="margin-top:0.5rem">
          <p style="font-size:0.78rem;color:#888;font-family:DM Mono,monospace;margin:0">{info['desc']}</p>
          <p style="font-size:0.75rem;margin:0.3rem 0 0">{info['speed']}</p>
        </div>""", unsafe_allow_html=True)
    else:
        selected_model = "Ensemble (Stacking)"
        st.markdown("""
        <div class="card card-accent" style="margin-top:0.5rem">
          <p style="font-size:0.78rem;color:#E8A838;font-family:DM Mono,monospace;margin:0">RF Meta-learner</p>
          <p style="font-size:0.75rem;margin:0.3rem 0 0">NB · LSTM · MiniLM · RoBERTa</p>
          <p style="font-size:0.7rem;color:#888;margin:0.2rem 0 0">F1 = 0.9931 on test set</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.72rem;color:#888;'
        'text-transform:uppercase;letter-spacing:0.1em">Options</p>',
        unsafe_allow_html=True
    )
    show_lime       = st.checkbox("LIME explanation", value=True)
    show_all_models = st.checkbox("Compare all models", value=False)
    if show_lime:
        lime_features = st.slider("Top features (LIME)", 5, 20, 10)
        lime_samples  = st.select_slider("LIME samples", options=[100, 200, 300, 500, 1000], value=300)
    else:
        lime_features, lime_samples = 10, 300

    st.markdown("---")
    if st.button("🔄 Reload models"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f'<p style="font-family:DM Mono,monospace;font-size:0.65rem;color:#555;line-height:1.6">'
        f'Keras: {_KERAS_BACKEND}<br>'
        f'LIME: {"✓ available" if _LIME_OK else "✗ " + _LIME_ERR}<br>'
        f'Device: {"🟢 GPU" if DEVICE.type == "cuda" else "🔵 CPU"} · {DEVICE}'
        f'</p>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
  <h1 style="font-family:'DM Serif Display',serif;font-size:2.4rem;margin:0;color:#E8E8E0">
    Job Posting Fraud <span style="color:#E8A838">Detector</span></h1>
  <p style="color:#888;font-size:0.9rem;margin:0.3rem 0 0;font-family:DM Mono,monospace">
    Paste a job posting · Choose a model · Get an instant verdict</p>
</div>""", unsafe_allow_html=True)

# ── Load models & status bar ──────────────────────────────────────────────────
with st.spinner("Loading models…"):
    M = load_all_models()

status_map = {
    "NB": "nb_error", "LSTM": "lstm_error", "MiniLM": "minilm_error",
    "RoBERTa": "roberta_error", "Ensemble": "ensemble_error",
}
cols = st.columns(5)
for col, (name, err_key) in zip(cols, status_map.items()):
    ok  = err_key not in M
    tip = M.get(err_key, "loaded OK")
    col.markdown(f"""
    <div style="text-align:center;background:#1E1E1E;border:1px solid #2A2A2A;
                border-radius:8px;padding:0.5rem;font-family:DM Mono,monospace;
                font-size:0.7rem" title="{tip}">
      {'🟢' if ok else '🔴'}
      <span style="color:{'#5dbb8a' if ok else '#e07070'}">{name}</span>
    </div>""", unsafe_allow_html=True)

# Show errors in collapsible (non-blocking)
load_errors = {n: M[k] for n, k in status_map.items() if k in M}
if load_errors:
    with st.expander("⚠ Load warnings — click to see details", expanded=False):
        for n, err in load_errors.items():
            st.error(f"**{n}**: {err}")

st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
col_in, col_out = st.columns([1.1, 0.9], gap="large")

with col_in:
    st.markdown('<p class="section-title">Job Posting Text</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload .txt", type=["txt"], label_visibility="collapsed")
    st.markdown('<p class="upload-hint">or paste text below</p>', unsafe_allow_html=True)

    default_text = (
        uploaded.read().decode("utf-8", errors="ignore")
        if uploaded
        else st.session_state.get("last_text", "")
    )
    job_text = st.text_area(
        "Posting", value=default_text, height=320,
        placeholder="Paste the full job posting here…",
        label_visibility="collapsed",
    )
    st.session_state["last_text"] = job_text

    st.markdown(
        '<p style="font-size:0.75rem;color:#888;font-family:DM Mono,monospace;'
        'margin:0.5rem 0 0.2rem">Quick examples:</p>', unsafe_allow_html=True
    )
    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button("🚨 Fraud example"):
            st.session_state["last_text"] = (
                "Work from home immediately, no experience needed. "
                "Earn $5000 weekly with zero investment required. "
                "We will send you a check — keep part and wire the rest back. "
                "Provide your bank account and ID to get started asap. "
                "Urgent hiring, apply now before positions are filled!"
            )
            st.rerun()
    with ec2:
        if st.button("✅ Legit example"):
            st.session_state["last_text"] = (
                "Software Engineer — Backend (Python) at Acme Corp. "
                "We are looking for a senior backend engineer with 3+ years of Python experience. "
                "Responsibilities include designing RESTful APIs and managing PostgreSQL databases. "
                "Competitive salary $90k–$120k, full health benefits, 401k matching. "
                "Apply via our careers portal at acmecorp.com/jobs."
            )
            st.rerun()

    analyse_btn = st.button("🔍 Analyse Posting", use_container_width=True)

with col_out:
    st.markdown('<p class="section-title">Analysis Result</p>', unsafe_allow_html=True)

    if not analyse_btn or not job_text.strip():
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 1rem">
          <p style="font-size:2.5rem;margin:0">🔍</p>
          <p style="color:#888;font-family:DM Mono,monospace;font-size:0.8rem;margin:0.5rem 0 0">
            Paste a job posting and click Analyse</p>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner(f"Running {selected_model}…"):
            prob = PREDICT_FNS[selected_model](job_text, M)

        if prob is None:
            err_key = {
                "Naive Bayes": "nb_error", "LSTM": "lstm_error",
                "MiniLM": "minilm_error", "RoBERTa": "roberta_error",
                "Ensemble (Stacking)": "ensemble_error",
            }.get(selected_model, "")
            err_msg = M.get(err_key, "Unknown error — check sidebar for details.")
            st.error(f"**{selected_model}** failed to run:\n\n`{err_msg}`")
        else:
            render_verdict(prob, selected_model)
            st.markdown("<div style='margin:0.8rem 0'></div>", unsafe_allow_html=True)
            prob_bar("Fraud probability",      prob,   is_fraud=True)
            prob_bar("Legitimate probability", 1-prob, is_fraud=False)

            wc   = len(job_text.split())
            conf = abs(prob - 0.5) * 2 * 100
            conf_lbl = "HIGH" if conf > 70 else "MED" if conf > 40 else "LOW"
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box">
                <div class="metric-val">{prob*100:.0f}%</div>
                <div class="metric-lbl">Fraud prob</div>
              </div>
              <div class="metric-box">
                <div class="metric-val">{wc}</div>
                <div class="metric-lbl">Word count</div>
              </div>
              <div class="metric-box">
                <div class="metric-val">{conf_lbl}</div>
                <div class="metric-lbl">Confidence</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── All-models comparison ─────────────────────────────────────────────────────
if analyse_btn and job_text.strip() and show_all_models:
    st.markdown('<p class="section-title">All Models Comparison</p>', unsafe_allow_html=True)
    scores = {}
    prog   = st.progress(0, text="Running all models…")
    for i, mn in enumerate(MODEL_INFO.keys()):
        p = PREDICT_FNS[mn](job_text, M)
        if p is not None:
            scores[mn] = p
        prog.progress((i + 1) / len(MODEL_INFO), text=f"✓ {mn}")
    prog.empty()

    if scores:
        st.pyplot(render_all_models_chart(scores), use_container_width=True)
        plt.close()
        rows = [
            {"Model": mn, "Fraud Prob": f"{p*100:.2f}%",
             "Verdict": "🚨 FRAUD" if p >= 0.5 else "✅ LEGIT",
             "Confidence": f"{abs(p-0.5)*2*100:.0f}%"}
            for mn, p in scores.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── LIME interpretability ─────────────────────────────────────────────────────
if analyse_btn and job_text.strip() and show_lime:
    st.markdown('<p class="section-title">🧠 Interpretability — LIME</p>', unsafe_allow_html=True)

    if not _LIME_OK:
        st.error(
            f"LIME is not importable in this Python environment.\n\n"
            f"Error: `{_LIME_ERR}`\n\n"
            f"Fix: make sure you activated the correct env before running Streamlit:\n"
            f"```\nconda activate nlp-proj\npip install lime\nstreamlit run app.py\n```"
        )
    else:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.8rem">
          <p style="font-size:0.8rem;color:#888;font-family:DM Mono,monospace;margin:0">
            LIME perturbs the input and trains a local linear model to explain
            <strong style="color:#E8A838">{selected_model}</strong>.&nbsp;
            <strong style="color:#e07070">Red = pushes toward FRAUD</strong>&nbsp;·&nbsp;
            <strong style="color:#5dbb8a">Green = pushes toward LEGIT</strong>
          </p>
        </div>""", unsafe_allow_html=True)

        with st.spinner(f"Running LIME ({lime_samples} samples)…"):
            lime_res = run_lime_cached(job_text, selected_model, lime_features, lime_samples)

        render_lime(lime_res)

        if selected_model == "Ensemble (Stacking)":
            if st.checkbox("Show LIME for each sub-model", value=False):
                for mn in ["Naive Bayes", "LSTM", "MiniLM", "RoBERTa"]:
                    with st.expander(f"LIME — {mn}", expanded=False):
                        with st.spinner(f"Running LIME for {mn}…"):
                            lr = run_lime_cached(job_text, mn, lime_features, lime_samples)
                        render_lime(lr)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #2A2A2A;
            text-align:center;font-family:DM Mono,monospace;font-size:0.68rem;color:#444">
  FraudLens · NB · LSTM · MiniLM · RoBERTa · RF Stacking Ensemble
</div>""", unsafe_allow_html=True)
