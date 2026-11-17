"""Shared engine: models, ATS scoring, Groq LLM, PDF/Word export. Used by the Employer and Candidate pages."""
import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # hide the harmless "UNEXPECTED" load report
from torch_geometric.nn import GATConv
from pypdf import PdfReader
from collections import Counter
from xml.sax.saxutils import escape
from dotenv import load_dotenv
import docx
import json
import time
import re
import io
import requests
import os

# ✅ PDF EXPORT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# -----------------------------
# DEVICE (CPU is fine - no GPU needed)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# GAT MODEL
# -----------------------------
class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)

def find_weights(name):
    # weights can be name.pth or name.pth.zip (same torch format)
    for candidate in (f"{name}.pth", f"{name}.pth.zip"):
        path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name}.pth not found in {BASE_DIR}")

# -----------------------------
# LOAD MODELS (cached - loaded only once)
# -----------------------------
@st.cache_resource(show_spinner="Loading DistilBERT + GAT models...")
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    embedder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    embedder.eval()

    gat_model = GATNet(768, 128, 64)
    gat_model.load_state_dict(torch.load(find_weights("gat_model"), map_location=device))
    gat_model.to(device)
    gat_model.eval()

    projection = nn.Linear(768, 64)
    projection.load_state_dict(torch.load(find_weights("projection"), map_location=device))
    projection.to(device)
    projection.eval()

    return tokenizer, embedder, gat_model, projection

tokenizer, embedder, gat_model, projection = load_models()

# -----------------------------
# GROQ API
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

@st.cache_data(ttl=3600, show_spinner=False)
def get_groq_models():
    # ask Groq which chat models this key can use (old llama models get retired)
    try:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=10)
        ids = [m["id"] for m in r.json()["data"] if m.get("active", True)]
        skip = ("whisper", "guard", "orpheus", "allam", "tts")
        chat = sorted(i for i in ids if not any(s in i for s in skip))
        return [m for m in PREFERRED_MODELS if m in chat] + [m for m in chat if m not in PREFERRED_MODELS] or PREFERRED_MODELS
    except Exception:
        return PREFERRED_MODELS

STYLES = {
    "Conservative": "Keep wording close to the original; only fix weak phrasing and add job keywords where the candidate clearly has that skill.",
    "Balanced": "Restructure and rewrite freely for impact and ATS keyword alignment, while staying truthful.",
    "Aggressive": "Maximise ATS alignment: mirror the job post's terminology wherever the candidate's real experience supports it and lead with the most relevant work.",
}

def groq_resume_feedback(resume_text, job_text, score, missing_keywords, model, style,
                         matched_keywords=(), retry_note=""):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an ATS resume expert.

Job Description:
{job_text}

Original Resume:
{resume_text}

ATS Score: {score:.1f}%
Job keywords the resume ALREADY matches (keep every one of these, word for word): {", ".join(matched_keywords) or "none"}
Job keywords missing from the resume: {", ".join(missing_keywords) or "none"}
Rewrite style: {STYLES[style]}
{retry_note}
RULES:
- Never invent employers, job titles, dates, degrees, certifications or metrics not in the original resume.
- You may rephrase, reorder and surface skills already evidenced in the resume using the job's exact terms.
- Keep all real content from the original (every job, project, skill, education entry) - rewrite it, don't drop it.
- Use the job post's exact wording for any missing keyword the candidate's experience truthfully supports.
- Skills the job needs that the candidate doesn't show go in "missing_skills", NOT in the resume.

Tasks:
1. Suggest improvements
2. Identify missing skills
3. Rewrite 3-5 bullet points professionally
4. If score < 80%, clearly say NOT A GOOD FIT and explain why
5. IMPORTANT: Generate a FULL improved version of the resume aligned with the job description.

Return ONLY valid JSON in this format:
{{
  "verdict": "STRONG FIT | GOOD FIT | PARTIAL FIT | NOT A GOOD FIT",
  "verdict_reason": "2-3 sentences",
  "strengths": ["..."],
  "improvements": ["..."],
  "missing_skills": ["..."],
  "rewritten_bullets": [{{"before": "...", "after": "..."}}],
  "updated_resume": "full resume as plain text: name on line 1, contact on line 2, UPPERCASE section headings, bullets start with '- '"
}}
"""

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "response_format": {"type": "json_object"}
    }

    for _ in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
        except requests.RequestException as e:
            return None, f"Could not reach Groq: {e}"
        if response.status_code != 429:
            break
        # free tier rate limit (tokens per minute) - wait as Groq asks, then retry
        wait = re.search(r"try again in ([\d.]+)s", response.text)
        time.sleep(min(float(wait.group(1)) + 1, 30) if wait else 10)

    if response.status_code == 200:
        try:
            return json.loads(response.json()['choices'][0]['message']['content']), None
        except json.JSONDecodeError:
            return None, "Model returned invalid JSON - try again or pick another model."
    else:
        return None, f"Error: {response.text}"

# -----------------------------
# PDF EXPORT
# -----------------------------
BULLET_RE = re.compile(r"^\s*([-*•●]|\d+[.)])\s+")

def is_heading(line):
    s = line.strip().rstrip(":")
    return 2 < len(s) <= 40 and s.upper() == s and any(c.isalpha() for c in s) and not BULLET_RE.match(line)

def create_pdf(text):
    # built-in PDF fonts lack fancy dashes/quotes/spaces -> they render as black boxes
    text = text.translate(str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                                         "‘": "'", "’": "'", "“": '"', "”": '"',
                                         " ": " ", " ": " ", " ": " ", "…": "..."}))
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Title"], fontSize=20, spaceAfter=2)
    contact_style = ParagraphStyle("Contact", parent=styles["Normal"], fontSize=9.5, alignment=1)
    head_style = ParagraphStyle("Head", parent=styles["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1f3a8a"), spaceBefore=8, spaceAfter=2)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=13.5)
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=14, bulletIndent=4)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)

    content = []
    lines = [l for l in text.replace("**", "").split("\n") if l.strip()]
    for i, line in enumerate(lines):
        safe = escape(line.strip())  # & < > would break reportlab
        if i == 0:
            content.append(Paragraph(safe, name_style))
        elif i == 1 and not is_heading(line):
            content.append(Paragraph(safe, contact_style))
        elif is_heading(line):
            content.append(Paragraph(escape(line.strip().rstrip(":")), head_style))
            content.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c7d2fe"), spaceAfter=3))
        elif BULLET_RE.match(line):
            content.append(Paragraph(escape(BULLET_RE.sub("", line).strip()), bullet_style, bulletText="•"))
        else:
            content.append(Paragraph(safe, body_style))
            content.append(Spacer(1, 2))

    doc.build(content)
    return buf.getvalue()

# -----------------------------
# WORD EXPORT
# -----------------------------
def create_docx(text):
    d = docx.Document()
    lines = [l for l in text.replace("**", "").split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if i == 0:
            d.add_heading(line.strip(), level=0)
        elif is_heading(line):
            d.add_heading(line.strip().rstrip(":"), level=2)
        elif BULLET_RE.match(line):
            d.add_paragraph(BULLET_RE.sub("", line).strip(), style="List Bullet")
        else:
            d.add_paragraph(line.strip())
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()

# -----------------------------
# EMBEDDING (chunked so long resumes aren't cut at 256 tokens)
# -----------------------------
def get_embedding(text, chunk_tokens=254, max_chunks=8):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] or [tokenizer.unk_token_id]
    chunks = [ids[i:i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)][:max_chunks]

    cls_vecs = []
    with torch.no_grad():
        for chunk in chunks:
            input_ids = torch.tensor([[tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]], device=device)
            output = embedder(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            cls_vecs.append(output.last_hidden_state[:, 0, :])

    return torch.cat(cls_vecs, dim=0).mean(dim=0, keepdim=True)

# -----------------------------
# RESUME READER (PDF / DOCX / TXT)
# -----------------------------
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(uploaded_file).paragraphs).strip()
    return uploaded_file.read().decode("utf-8", errors="ignore").strip()

# -----------------------------
# KEYWORDS (stopwords + job-post filler removed)
# -----------------------------
STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before being below between both but by can
could did do does doing down during each etc few for from further had has have having he her here him his how i if in
into is it its just me more most my no nor not now of off on once only or other our out over own per same she should
so some such than that the their them then there these they this those through to too under until up us very via was
we were what when where which while who whom why will with within without would you your
ability able across applicant apply based benefits best candidate career closely company competitive culture day
demonstrated desired duties environment equal excellent experience experienced familiar familiarity field fast full
good great help high highly ideal including job join key knowledge least like looking make must new nice offer
opportunity paced part plus position preferred previous proven qualifications related required requirements
responsibilities responsible role salary seeking skills strong successful support team teams time understanding using
well work working year years
build building design designing develop developing maintain write writing collaborate ensure create manage
implement deliver drive engineer engineers developer developers product managers development services
""".split())

def is_junk(token):
    # links, hashtags, ids, mixed digits+letters like "12thpass" - no resume can match these
    return (token.startswith(("http", "www")) or ("#" in token and token not in ("c#", "f#"))
            or re.search(r"\.(com|in|io|ly|org|net)\b|/\w", token) is not None
            or re.fullmatch(r"\d+\+?|\d+[a-z]{2,}\w*", token) is not None or len(token) > 25)

def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text.lower())  # drop URLs / emails
    tokens = re.findall(r"[a-z0-9][a-z0-9+#.\-/]*[a-z0-9+#]|[a-z0-9]", text)
    return [t for t in tokens if t not in STOPWORDS and not is_junk(t) and len(t) > 1]

def keyword_match(resume_text, job_text, top_n=40):
    job_keywords = [w for w, _ in Counter(clean_text(job_text)).most_common(top_n)]
    resume_words = set(clean_text(resume_text))

    matched = [k for k in job_keywords if k in resume_words or k.rstrip("s") in resume_words or k + "s" in resume_words]
    missing = [k for k in job_keywords if k not in matched]
    keyword_score = len(matched) / max(len(job_keywords), 1)
    return keyword_score, matched, missing

# -----------------------------
# ATS SCORE (GAT + projection + keywords)
# -----------------------------
PROJ_LOW, PROJ_HIGH = 0.90, 0.99  # projection cosine range seen from unrelated -> strong match
# (wide enough that small wording changes in a rewrite don't swing the score by several points)

def ats_score(resume_text, job_text):
    resume_emb = get_embedding(resume_text)
    job_emb = get_embedding(job_text)

    x = torch.cat([resume_emb, job_emb], dim=0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)

    with torch.no_grad():
        x_out = gat_model(x, edge_index)
        proj_out = projection(x)

    # NOTE: with a 2-node graph both GAT outputs come out identical (cosine = 1.0 for ANY pair),
    # so the GAT value is shown but not used. The projection layer does separate good/bad matches,
    # but only within ~0.92-0.985 cosine, so it is rescaled to 0-1 over that band.
    gat_score = F.cosine_similarity(x_out[0:1], x_out[1:2]).item()
    proj_score = F.cosine_similarity(proj_out[0:1], proj_out[1:2]).item()
    semantic_score = max(0, min((proj_score - PROJ_LOW) / (PROJ_HIGH - PROJ_LOW), 1))

    keyword_score, matched, missing = keyword_match(resume_text, job_text)

    final_score = (0.6 * semantic_score) + (0.4 * keyword_score)
    return {
        "score": max(0, min(final_score * 100, 100)),
        "semantic": semantic_score * 100,
        "gat": gat_score * 100,
        "projection": proj_score * 100,
        "keyword": keyword_score * 100,
        "matched": matched,
        "missing": missing,
    }

# -----------------------------
# SUMMARY
# -----------------------------
def generate_summary(resume_text):
    words = resume_text.split()
    return " ".join(words[:80]) + ("..." if len(words) > 80 else "")

