# ❓ Automatic Question Generator

A web application that automatically generates meaningful questions from BBC news articles using NLP — built with spaCy, NLTK, and Streamlit.

---

## 🗂️ Project Structure

```
auto_question_generator/
│
├── app.py                   # Streamlit web application
├── BBC_news_dataset.csv     # Dataset (place in same folder)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone / Download the project

```bash
mkdir auto_question_generator
cd auto_question_generator
# Copy app.py, requirements.txt, and BBC_news_dataset.csv here
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK stopwords

```python
python -c "import nltk; nltk.download('stopwords')"
```

### 6. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free — Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit — Automatic Question Generator"
   git remote add origin https://github.com/YOUR_USERNAME/auto-question-generator.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. Click **"New app"** → Connect your GitHub repo

4. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10

5. Add a `packages.txt` file for the spaCy model:
   ```
   # packages.txt  (add to root of your repo)
   ```
   And a `setup.sh`:
   ```bash
   #!/bin/bash
   python -m spacy download en_core_web_sm
   ```

6. Click **Deploy** — your app goes live at `https://YOUR_APP.streamlit.app`

---

## 🐳 Deploy with Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t question-generator .
docker run -p 8501:8501 question-generator
```

---

## 🖥️ Deploy on Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Upload `app.py`, `requirements.txt`, and `BBC_news_dataset.csv`
4. The app deploys automatically

---

## 🔬 How It Works

### Question Generation Pipeline

```
Raw Text
   │
   ▼
spaCy NLP Processing
   │
   ├── Dependency Parsing  → Who/What/Did questions
   │     (nsubj + ROOT verb)
   │
   └── Named Entity Recognition (NER)
         ├── PERSON → "Who is X?"
         ├── ORG    → "What is X?"
         ├── GPE    → "Where is X mentioned?"
         ├── DATE   → "What happened on X?"
         └── MONEY  → "What does the figure X represent?"
```

### Text Classification (from Notebook)

| Approach | Method | Notes |
|----------|--------|-------|
| Deep Learning | Bidirectional LSTM | Embedding → BiLSTM → Dense |
| ML | TF-IDF + Random Forest | n_estimators=100 |

### Dataset

| Field | Description |
|-------|-------------|
| `description` | Full news article text |
| `tags` | Comma-separated topic tags |
| `label` | First tag = category |

**2,410 articles** across 6 main categories: Sports, Entertainment, Business, Technology, Politics, Law.

---

## 📦 Key Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `spacy` | NLP — dependency parsing & NER |
| `nltk` | Stopword removal |
| `pandas` | Data handling |
| `scikit-learn` | ML classification (TF-IDF + RF) |
| `tensorflow` | Deep learning (Bi-LSTM) |
| `wordcloud` | EDA visualization |

---

## 🛠️ Troubleshooting

**`spacy model not found`**
```bash
python -m spacy download en_core_web_sm
```

**`NLTK stopwords not found`**
```python
import nltk; nltk.download('stopwords')
```

**`Dataset not found`**
Ensure `BBC_news_dataset.csv` is in the same folder as `app.py`.

---

## 📄 License
MIT License — free to use and modify.
