# 🎯 ResumeMatch AI

**AI-Based Resume–Job Description Matching System**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-success?style=for-the-badge)](https://kaif-resumematch.streamlit.app)

An intelligent NLP-powered tool that analyzes how well your resume matches a
job description, giving you a detailed score, skills gap analysis, ATS keyword
coverage, and actionable improvement suggestions.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Match Score (0–100)** | Weighted score combining skills, experience, keywords, education |
| **Skills Analysis** | Matched ✅, missing ❌, and extra ➕ skills |
| **ATS Keyword Analysis** | Which JD keywords appear / are missing in your resume |
| **Experience Alignment** | Compares years required vs years on resume |
| **Education Check** | Detects degree requirements and verifies against resume |
| **AI Suggestions** | Personalized, actionable tips to improve your resume |
| **Visual Dashboard** | Score circle, progress bars, color-coded skill tags |

---

## 🗂️ Project Structure

```
resume_matcher/
├── app.py                  # Streamlit UI (main entry point)
├── requirements.txt        # Python dependencies
├── README.md
├── backend/
│   ├── __init__.py
│   ├── parser.py           # PDF extraction + text cleaning
│   ├── matcher.py          # NLP matching engine (skills, ATS, similarity)
│   └── scorer.py           # Weighted scoring + AI suggestions
└── utils/
    ├── __init__.py
    └── text_cleaner.py     # Reusable text utilities
```

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **NLP Model**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Similarity**: Cosine similarity via scikit-learn
- **PDF Parsing**: pdfplumber (+ PyPDF2 fallback)
- **Language**: Python 3.10+

---

## 🚀 How to Run

### 1. Prerequisites

- Python 3.10 or higher
- pip

### 2. Clone / Download the project

```bash
git clone <your-repo-url>
cd resume_matcher
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The `sentence-transformers` package downloads the `all-MiniLM-L6-v2`
> model (~80 MB) on first run. Ensure you have an internet connection the first time.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open locally in your browser.

---

## 📋 How to Use

1. **Upload Resume**: Click the upload area and select your PDF resume.
2. **Paste Job Description**: Copy the full JD text into the text box.
3. **Click "Analyze Match"**: Wait a few seconds for the NLP analysis.
4. **Review Results**:
   - **Overall Score** – your match percentage with a verdict
   - **Score Breakdown** – per-category progress bars
   - **Skills Analysis** – matched, missing, and extra skills
   - **ATS Keywords** – coverage of critical JD keywords
   - **Experience & Education** – alignment with JD requirements
   - **AI Suggestions** – specific steps to improve your resume

---

## 📊 Scoring Weights

| Category | Weight |
|---|---|
| Skills Match | **40%** |
| Experience Alignment | **30%** |
| ATS Keywords | **20%** |
| Education Match | **10%** |
| Semantic Similarity | Bonus (+5 max) |

### Score Interpretation

| Score | Verdict |
|---|---|
| 80–100 | 🟢 Excellent Match |
| 60–79 | 🔵 Good Match |
| 40–59 | 🟡 Moderate Match |
| 0–39 | 🔴 Weak Match |

---

## 🧪 Sample Input / Output

### Input
- **Resume**: PDF resume of a software engineer with Python, Django, PostgreSQL, Docker skills, 3 years experience, B.Tech degree.
- **JD**: Senior Python Developer role requiring Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, 5+ years experience, Bachelor's degree.

### Expected Output
```
Overall Score: 62 / 100  →  Good Match

Skills Breakdown:
  ✅ Matched Skills (5):  python, postgresql, docker, django, git
  ❌ Missing Skills (4):  fastapi, redis, kubernetes, aws
  ➕ Extra Skills (3):    html, css, javascript

Score Breakdown:
  Skills Match:    58.0% ████████░░
  Experience:      60.0% ██████░░░░   (3yr actual vs 5yr required)
  ATS Keywords:    65.0% ██████░░░░
  Education:      100.0% ██████████

AI Suggestions:
  💡 Add these missing skills: 'fastapi', 'redis', 'kubernetes', 'aws'
  💡 The job requires 5+ years but resume shows ~3 years...
  💡 Include ATS keywords: 'Senior Python', 'AWS', 'Kubernetes'...
  💡 Quantify your achievements wherever possible...
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: pdfplumber` | Run `pip install pdfplumber` |
| `No text extracted from PDF` | PDF may be scanned/image-based. Try a text-based PDF. |
| Model download slow | First run downloads ~80MB model. Wait for completion. |
| `torch` install issues | Install CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

---

## 📝 License

MIT License – free to use and modify.
