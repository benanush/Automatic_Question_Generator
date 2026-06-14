"""
Automatic Question Generator - Streamlit Web App
Based on BBC News Dataset + spaCy NLP Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Automatic Question Generator",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .question-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.95rem;
        color: #1e293b;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    }
    .question-card.who   { border-left-color: #7c3aed; background: linear-gradient(135deg, #f5f0ff 0%, #ede9fe 100%); }
    .question-card.when  { border-left-color: #0891b2; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); }
    .question-card.where { border-left-color: #16a34a; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); }
    .question-card.what  { border-left-color: #ea580c; background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); }
    .question-card.how   { border-left-color: #db2777; background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); }

    .entity-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    .badge-org   { background:#e0e7ff; color:#3730a3; }
    .badge-gpe   { background:#dcfce7; color:#15803d; }
    .badge-date  { background:#dbeafe; color:#1d4ed8; }
    .badge-per   { background:#fce7f3; color:#9d174d; }
    .badge-num   { background:#fef9c3; color:#854d0e; }
    .badge-other { background:#f1f5f9; color:#475569; }

    .stat-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #4f46e5; }
    .stat-label  { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }

    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 4px;
        border-bottom: 2px solid #e5e7eb;
    }
    .category-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .upload-box {
        border: 2px dashed #c7d2fe;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: #f5f7ff;
        margin: 10px 0;
    }
    .upload-hint {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 6px;
        line-height: 1.5;
    }
    .col-map-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
    }
    .success-badge {
        background: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .warning-badge {
        background: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load NLP Resources ──────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

nlp        = load_nlp()
stop_words = load_stopwords()

# ─── Dataset Loader from Uploaded File ──────────────────────────────────────
def load_uploaded_dataset(uploaded_file, text_col, tag_col):
    """Load and process any uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        if text_col not in df.columns:
            return None, f"Column '{text_col}' not found in uploaded file."
        df["_text"]  = df[text_col].astype(str)
        df["_label"] = (
            df[tag_col].astype(str).apply(lambda x: x.split(",")[0].strip())
            if tag_col and tag_col in df.columns
            else "general"
        )
        df["_word_count"] = df["_text"].apply(lambda x: len(x.split()))
        return df, None
    except Exception as e:
        return None, str(e)

def guess_text_col(columns):
    """Auto-detect the text column from common names."""
    priority = ["description", "text", "content", "article", "body", "news", "sentence"]
    for p in priority:
        for c in columns:
            if p in c.lower():
                return c
    return columns[0]

def guess_tag_col(columns):
    """Auto-detect the tag/category column."""
    priority = ["tags", "tag", "category", "label", "class", "topic", "genre"]
    for p in priority:
        for c in columns:
            if p in c.lower():
                return c
    return None

# ─── Helper Functions ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text  = str(text).lower()
    text  = re.sub(r"[^a-zA-Z ]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

CATEGORY_COLORS = {
    "sports":        ("#dcfce7", "#15803d"),
    "entertainment": ("#fce7f3", "#9d174d"),
    "business":      ("#dbeafe", "#1d4ed8"),
    "technology":    ("#e0e7ff", "#3730a3"),
    "politics":      ("#fff7ed", "#c2410c"),
    "law":           ("#fef9c3", "#854d0e"),
}

def get_category_style(label: str):
    label = label.lower()
    for k, (bg, fg) in CATEGORY_COLORS.items():
        if k in label:
            return bg, fg
    return "#f1f5f9", "#475569"

def classify_question_type(q: str) -> str:
    q = q.lower()
    if q.startswith("who"):   return "who"
    if q.startswith("when"):  return "when"
    if q.startswith("where"): return "where"
    if q.startswith("what"):  return "what"
    if q.startswith("how"):   return "how"
    return ""

def generate_questions(text: str):
    doc       = nlp(text)
    questions = []

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ in ("VERB", "AUX"):
            subject = token.text.capitalize()
            verb    = token.head.lemma_
            if token.dep_ == "nsubjpass":
                questions.append(f"What was {subject} {verb}ed?")
            else:
                obj = next((c for c in token.head.children if c.dep_ in ("dobj", "attr", "pobj")), None)
                if obj:
                    questions.append(f"What did {subject} {verb}?")
                else:
                    questions.append(f"What did {subject} do?")

    for ent in doc.ents:
        txt = ent.text.strip()
        if not txt or len(txt) < 2:
            continue
        if ent.label_ == "PERSON":
            questions.append(f"Who is {txt}?")
        elif ent.label_ in ("ORG", "FAC", "PRODUCT"):
            questions.append(f"What is {txt}?")
        elif ent.label_ in ("GPE", "LOC"):
            questions.append(f"Where is {txt} mentioned in this context?")
        elif ent.label_ == "DATE":
            questions.append(f"What happened on {txt}?")
        elif ent.label_ == "TIME":
            questions.append(f"What occurred at {txt}?")
        elif ent.label_ in ("MONEY", "PERCENT", "QUANTITY", "CARDINAL"):
            questions.append(f"What does the figure '{txt}' represent?")
        elif ent.label_ == "EVENT":
            questions.append(f"What is the significance of {txt}?")
        elif ent.label_ == "LAW":
            questions.append(f"What does {txt} refer to?")

    for sent in doc.sents:
        root = [t for t in sent if t.dep_ == "ROOT"]
        if root and root[0].pos_ == "VERB":
            subj = next((t for t in root[0].lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
            if subj:
                questions.append(f"Did {subj.text} {root[0].lemma_}?")

    seen   = set()
    unique = []
    for q in questions:
        norm = q.lower().strip()
        if norm not in seen and len(q) > 8:
            seen.add(norm)
            unique.append(q)
    return unique[:20]

def extract_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def guess_category(text: str) -> str:
    text  = text.lower()
    rules = {
        "sports":        ["match", "goal", "team", "player", "game", "season", "club", "win", "score", "football", "cricket", "tennis"],
        "technology":    ["software", "technology", "device", "internet", "digital", "ai", "computer", "app", "data", "cyber"],
        "business":      ["company", "market", "economy", "trade", "profit", "share", "stock", "bank", "invest", "revenue"],
        "politics":      ["government", "minister", "election", "parliament", "policy", "political", "party", "vote", "law", "court"],
        "entertainment": ["film", "music", "actor", "award", "celebrity", "concert", "movie", "album", "show", "star"],
        "law":           ["court", "judge", "sentence", "crime", "arrest", "police", "lawyer", "case", "trial", "legal"],
    }
    scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in rules.items()}
    top    = max(scores, key=scores.get)
    return top if scores[top] > 0 else "general"

# ─── Session State Init ──────────────────────────────────────────────────────
if "df"        not in st.session_state: st.session_state.df        = None
if "text_col"  not in st.session_state: st.session_state.text_col  = None
if "tag_col"   not in st.session_state: st.session_state.tag_col   = None
if "file_name" not in st.session_state: st.session_state.file_name = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    max_questions = st.slider("Max Questions", 3, 20, 10)
    show_entities = st.checkbox("Show Named Entities", value=True)
    show_stats    = st.checkbox("Show Text Stats",    value=True)

    st.markdown("---")

    # ── Dataset Upload ────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload any CSV with a text/description column. Supports BBC News format and custom datasets."
    )

    if uploaded_file is not None:
        # Only re-process if a new file was uploaded
        if uploaded_file.name != st.session_state.file_name:
            # Peek at columns to auto-detect
            try:
                peek = pd.read_csv(uploaded_file, nrows=1)
                uploaded_file.seek(0)   # rewind after peek
                cols     = peek.columns.tolist()
                auto_txt = guess_text_col(cols)
                auto_tag = guess_tag_col(cols)
                st.session_state._cols     = cols
                st.session_state._auto_txt = auto_txt
                st.session_state._auto_tag = auto_tag
                st.session_state._raw_file = uploaded_file
                st.session_state.file_name = uploaded_file.name
                st.session_state.df        = None  # reset until columns confirmed
            except Exception as e:
                st.error(f"Could not read file: {e}")

        # Column mapping UI
        if st.session_state.file_name == uploaded_file.name and hasattr(st.session_state, "_cols"):
            st.markdown('<div class="col-map-box">', unsafe_allow_html=True)
            st.markdown("**Map your columns**")

            cols        = st.session_state._cols
            none_option = ["— None —"]

            text_col = st.selectbox(
                "📝 Text column",
                cols,
                index=cols.index(st.session_state._auto_txt) if st.session_state._auto_txt in cols else 0,
                key="sel_text"
            )
            tag_col_options = none_option + cols
            default_tag_idx = (
                tag_col_options.index(st.session_state._auto_tag)
                if st.session_state._auto_tag and st.session_state._auto_tag in tag_col_options
                else 0
            )
            tag_col_sel = st.selectbox(
                "🏷️ Category/Tags column (optional)",
                tag_col_options,
                index=default_tag_idx,
                key="sel_tag"
            )
            tag_col = None if tag_col_sel == "— None —" else tag_col_sel
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("✅ Load Dataset", use_container_width=True):
                uploaded_file.seek(0)
                df, err = load_uploaded_dataset(uploaded_file, text_col, tag_col)
                if err:
                    st.error(err)
                else:
                    st.session_state.df       = df
                    st.session_state.text_col = text_col
                    st.session_state.tag_col  = tag_col
                    st.success(f"Loaded {len(df):,} rows!")

    # Dataset stats (once loaded)
    df = st.session_state.df
    if df is not None:
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Articles", f"{len(df):,}")
        if st.session_state.tag_col:
            st.metric("Categories", df["_label"].nunique())
            cat_counts = df["_label"].value_counts().head(5)
            st.markdown("**Top Categories**")
            for cat, cnt in cat_counts.items():
                bg, fg = get_category_style(cat)
                st.markdown(
                    f'<span class="category-pill" style="background:{bg};color:{fg}">'
                    f'{cat.title()} ({cnt})</span>',
                    unsafe_allow_html=True
                )
        st.markdown(f'<span class="success-badge">✓ {st.session_state.file_name}</span>', unsafe_allow_html=True)
    elif uploaded_file is None:
        st.markdown(
            '<p class="upload-hint">⬆️ Upload a CSV above to enable <b>Pick from Dataset</b> mode.<br>'
            'You can still use <b>Enter Custom Text</b> mode without uploading.</p>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Generates questions from news text using spaCy dependency parsing "
        "and named entity recognition (NER)."
    )

# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">❓ Automatic Question Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transform any news article into insightful questions using NLP</div>', unsafe_allow_html=True)

df = st.session_state.df

# Input mode
mode = st.radio("Input Mode", ["✏️ Enter Custom Text", "📰 Pick from Dataset"], horizontal=True)

input_text = ""

if mode == "✏️ Enter Custom Text":
    input_text = st.text_area(
        "Paste your news article or text here:",
        height=180,
        placeholder="e.g. The prime minister announced new economic policies at Parliament on Monday..."
    )

else:  # Pick from Dataset
    if df is not None:
        has_categories = st.session_state.tag_col is not None
        if has_categories:
            categories = ["All"] + sorted(df["_label"].unique().tolist())
            sel_cat    = st.selectbox("Filter by Category", categories)
            subset     = df if sel_cat == "All" else df[df["_label"] == sel_cat]
        else:
            subset = df

        idx = st.selectbox(
            f"Choose Article ({len(subset):,} available)",
            subset.index,
            format_func=lambda i: f"#{i} — {str(subset.loc[i, '_text'])[:80]}…"
        )

        input_text = str(df.loc[idx, "_text"])

        if has_categories:
            cat_label = str(df.loc[idx, "_label"])
            bg, fg    = get_category_style(cat_label)
            st.markdown(
                f'Category: <span class="category-pill" style="background:{bg};color:{fg}">'
                f'{cat_label.title()}</span>',
                unsafe_allow_html=True
            )

        with st.expander("📄 Full Article Text"):
            st.write(input_text)

    else:
        st.info(
            "📂 No dataset loaded yet. Upload a CSV file in the **sidebar** to use this mode.\n\n"
            "Accepted format: any CSV with at least one text column.\n"
            "Example: `description`, `text`, `article`, `content` columns all work."
        )
        st.markdown(
            "**Expected CSV format (example):**\n"
            "```\ndescription, tags\n"
            "Chelsea sacked Adrian Mutu..., sports, football\n"
            "The prime minister announced..., politics, UK\n```"
        )

# ─── Generate Button ──────────────────────────────────────────────────────────
if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text or pick an article from the dataset first.")
    else:
        with st.spinner("Analysing text and generating questions…"):
            questions  = generate_questions(input_text)[:max_questions]
            entities   = extract_entities(input_text)
            guessed    = guess_category(input_text)
            word_count = len(input_text.split())
            sent_count = len(list(nlp(input_text).sents))

        st.markdown("---")

        if show_stats:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{len(questions)}</div><div class="stat-label">Questions</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{word_count}</div><div class="stat-label">Words</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{sent_count}</div><div class="stat-label">Sentences</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{len(entities)}</div><div class="stat-label">Entities</div></div>', unsafe_allow_html=True)
            st.markdown("")

        col_q, col_e = st.columns([3, 2])

        with col_q:
            st.markdown('<div class="section-header">🔹 Generated Questions</div>', unsafe_allow_html=True)
            if questions:
                for i, q in enumerate(questions, 1):
                    qtype = classify_question_type(q)
                    st.markdown(
                        f'<div class="question-card {qtype}"><b>Q{i}.</b> {q}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No questions generated. Try a longer or more structured text.")

        with col_e:
            if show_entities and entities:
                st.markdown('<div class="section-header">🏷️ Named Entities</div>', unsafe_allow_html=True)
                badge_map = {
                    "ORG": "badge-org",  "GPE": "badge-gpe", "LOC": "badge-gpe",
                    "DATE": "badge-date", "TIME": "badge-date",
                    "PERSON": "badge-per", "NORP": "badge-per",
                    "CARDINAL": "badge-num", "MONEY": "badge-num", "PERCENT": "badge-num",
                }
                badges_html = ""
                for txt, label in entities[:25]:
                    cls = badge_map.get(label, "badge-other")
                    badges_html += f'<span class="entity-badge {cls}" title="{label}">{txt}</span>'
                st.markdown(badges_html, unsafe_allow_html=True)

            st.markdown('<div class="section-header">🗂️ Predicted Category</div>', unsafe_allow_html=True)
            bg, fg = get_category_style(guessed)
            st.markdown(
                f'<span class="category-pill" style="background:{bg};color:{fg};font-size:1rem;padding:6px 18px">'
                f'{guessed.title()}</span>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-header">📊 Question Types</div>', unsafe_allow_html=True)
            types       = [classify_question_type(q) or "other" for q in questions]
            type_counts = Counter(types)
            for qt, cnt in type_counts.most_common():
                pct = int(cnt / len(types) * 100) if types else 0
                st.progress(pct / 100, text=f"{qt.upper() if qt else 'OTHER'} — {cnt}")

        st.markdown("---")
        questions_txt = "\n".join(f"Q{i+1}. {q}" for i, q in enumerate(questions))
        st.download_button(
            "⬇️ Download Questions (.txt)",
            data=questions_txt,
            file_name="generated_questions.txt",
            mime="text/plain"
        )