# ATS Resume Optimizer with LLM

Streamlit app for both sides of hiring:

- **🏢 Employer:** upload many resumes at once (PDF / DOCX / TXT or a ZIP), score every one against the job description, set a minimum ATS score and must-have skills, and get a ranked shortlist of eligible candidates. Download the shortlist as CSV and the eligible resumes as a ZIP.
- **👤 Candidate:** score your resume against a job post and let the LLM (Groq) redesign it for that job, then compare old vs new score and download it.

## Files

| File | What it is |
|---|---|
| `app.py` | Entry point: page navigation |
| `views/home.py` | Home page |
| `views/employer.py` | Employer bulk screening |
| `views/candidate.py` | Candidate resume upgrade |
| `ats_core.py` | Shared models, ATS scoring, Groq LLM, PDF/Word export |

## How it works

- **Scoring:** DistilBERT embeddings → trained GAT + projection layer (`gat_model.pth.zip`, `projection.pth.zip`) for semantic match, plus job-keyword coverage.
- **Upgrade:** Groq LLM gives feedback, missing skills, bullet rewrites and a full tailored resume. The new resume is re-scored, and the app retries (up to 3 times) to keep the best version.
- **Export:** PDF, Word or TXT.
- Runs on **CPU**; no GPU needed.

## Setup (Windows)

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Groq API key in it (get one at https://console.groq.com/keys):

```
GROQ_API_KEY=your_groq_api_key_here
```

## Run

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Or double-click `run.bat`. Open http://localhost:8501.

The first run downloads DistilBERT (~260 MB). The model weight files are loaded directly from the `.pth.zip` files; don't unzip them.
