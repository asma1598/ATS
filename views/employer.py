import streamlit as st
st.title("🏢 Employer · Bulk Resume Screening")

from ats_core import *
import zipfile
import pandas as pd

st.caption(f"Upload many resumes, score them all against your job post, and keep only the eligible ones · DistilBERT + GAT on **{device.type.upper()}**")

RESUME_TYPES = (".pdf", ".docx", ".txt")

# -----------------------------
# HELPERS
# -----------------------------
class NamedBytes(io.BytesIO):
    # lets files pulled out of a ZIP go through extract_text_from_file like normal uploads
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

def collect_resumes(uploaded_files):
    # flatten uploads: single resumes + every resume inside any ZIP
    files = []
    for f in uploaded_files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if info.is_dir() or base.startswith((".", "~$")) or not base.lower().endswith(RESUME_TYPES):
                        continue
                    files.append((base, z.read(info)))
        else:
            files.append((f.name, f.getvalue()))
    return files

def candidate_name(text, filename):
    # first short line that looks like a name, else the file name
    for line in text.splitlines()[:5]:
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and len(line) <= 40 and not re.search(r"[@\d|:/]", line):
            return line.title()
    return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()

def find_email(text):
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return m.group(0) if m else ""

def find_phone(text):
    m = re.search(r"(\+?\d[\d\s\-()]{8,}\d)", text)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

def has_skill(skill, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])", text.lower()) is not None

def screen(files, job_text, must_have, min_score):
    rows, unreadable = [], []
    progress = st.progress(0.0, text="Screening resumes...")
    for i, (name, data) in enumerate(files, 1):
        progress.progress(i / len(files), text=f"Screening {i}/{len(files)}: {name}")
        try:
            text = extract_text_from_file(NamedBytes(data, name))
        except Exception as e:
            unreadable.append({"File": name, "Reason": f"Could not read file ({type(e).__name__})"})
            continue
        if not text:
            unreadable.append({"File": name, "Reason": "No text found (scanned image PDF?)"})
            continue

        result = ats_score(text, job_text)
        have = [s for s in must_have if has_skill(s, text)]
        lacking = [s for s in must_have if s not in have]

        reasons = []
        if result["score"] < min_score:
            reasons.append(f"Score below {min_score}%")
        if lacking:
            reasons.append("Missing: " + ", ".join(lacking))

        rows.append({
            "Candidate": candidate_name(text, name),
            "Email": find_email(text),
            "Phone": find_phone(text),
            "ATS Score": round(result["score"], 2),
            "Semantic Match": round(result["semantic"], 1),
            "Must-have": f"{len(have)}/{len(must_have)}" if must_have else "-",
            "Eligible": not reasons,
            "Reason": "; ".join(reasons) or "Meets all requirements",
            "File": name,
            "_text": text,
            "_data": data,
        })
    progress.empty()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ATS Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df, pd.DataFrame(unreadable)

def zip_files(df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for _, r in df.iterrows():
            z.writestr(f"{r['Rank']:03d}_{r['File']}", r["_data"])
    return buf.getvalue()

PUBLIC_COLS = ["Rank", "Candidate", "Email", "Phone", "ATS Score", "Semantic Match", "Must-have", "Reason", "File"]

# -----------------------------
# UI
# -----------------------------
col_left, col_right = st.columns(2)
with col_left:
    job_text = st.text_area("Paste Job Description", height=260, key="emp_job")
with col_right:
    uploads = st.file_uploader("Upload Resumes (PDF / DOCX / TXT, or a ZIP of them)",
                               type=["pdf", "docx", "txt", "zip"], accept_multiple_files=True, key="emp_files")
    must_have_raw = st.text_input("Must-have skills (comma separated, optional)",
                                  placeholder="e.g. Python, SQL, Docker", key="emp_must")
    min_score = st.slider("Minimum ATS score to be eligible", 0, 100, 80, key="emp_min")

screen_btn = st.button("🔍 Screen All Resumes", type="primary", width="stretch")

if screen_btn:
    if not uploads or job_text.strip() == "":
        st.warning("Please upload resumes and enter the job description")
        st.stop()

    files = collect_resumes(uploads)
    if not files:
        st.warning("No PDF / DOCX / TXT resumes found in the upload")
        st.stop()

    must_have = [s.strip() for s in must_have_raw.split(",") if s.strip()]
    df, bad = screen(files, job_text, must_have, min_score)
    st.session_state["emp_df"] = df
    st.session_state["emp_bad"] = bad

# -----------------------------
# RESULTS
# -----------------------------
if "emp_df" in st.session_state:
    df, bad = st.session_state["emp_df"], st.session_state["emp_bad"]
    eligible = df[df["Eligible"]] if not df.empty else df
    rejected = df[~df["Eligible"]] if not df.empty else df

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes screened", len(df) + len(bad))
    m2.metric("✅ Eligible", len(eligible))
    m3.metric("❌ Not eligible", len(rejected))
    m4.metric("⚠️ Unreadable", len(bad))

    tab1, tab2, tab3 = st.tabs([f"✅ Eligible ({len(eligible)})", f"❌ Not eligible ({len(rejected)})",
                                f"⚠️ Unreadable ({len(bad)})"])

    with tab1:
        if eligible.empty:
            st.info("No candidate meets the requirements. Try lowering the minimum score or the must-have skills.")
        else:
            st.dataframe(eligible[PUBLIC_COLS], hide_index=True, width="stretch",
                         column_config={"ATS Score": st.column_config.ProgressColumn("ATS Score", format="%.2f%%", min_value=0, max_value=100)})

            d1, d2 = st.columns(2)
            d1.download_button("📥 Download shortlist (CSV)", eligible[PUBLIC_COLS].to_csv(index=False),
                               "eligible_candidates.csv", "text/csv", width="stretch")
            d2.download_button("📦 Download eligible resumes (ZIP)", zip_files(eligible),
                               "eligible_resumes.zip", "application/zip", width="stretch")

            st.markdown("**Candidate profiles**")
            for _, r in eligible.iterrows():
                with st.expander(f"#{r['Rank']} · {r['Candidate']} · {r['ATS Score']:.2f}%"):
                    st.write(f"📧 {r['Email'] or '-'}  ·  📞 {r['Phone'] or '-'}  ·  📄 {r['File']}")
                    st.write(generate_summary(r["_text"]))

    with tab2:
        if rejected.empty:
            st.info("Every readable resume is eligible.")
        else:
            st.dataframe(rejected[PUBLIC_COLS], hide_index=True, width="stretch")

    with tab3:
        if bad.empty:
            st.info("All files were read successfully.")
        else:
            st.dataframe(bad, hide_index=True, width="stretch")

    if not df.empty:
        st.download_button("📥 Download full report (all candidates, CSV)", df[PUBLIC_COLS + ["Eligible"]].to_csv(index=False),
                           "screening_report.csv", "text/csv")
