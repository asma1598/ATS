import streamlit as st
st.title("👤 Candidate · Resume Upgrade")

from ats_core import *
st.caption(f"DistilBERT + GAT scoring on **{device.type.upper()}** · Groq AI resume upgrade")

with st.sidebar:
    st.header("⚙️ Settings")
    groq_model = st.selectbox("Groq model", get_groq_models())
    style = st.radio("Rewrite style", list(STYLES), index=1)
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing - add it to the .env file")

col_left, col_right = st.columns(2)
with col_left:
    resume_file = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
with col_right:
    job_text = st.text_area("Paste Job Description", height=220)

b1, b2 = st.columns(2)
analyze_btn = b1.button("🔍 Analyze Resume", width="stretch")
ai_btn = b2.button("✨ Get AI Feedback + Upgrade Resume", type="primary", width="stretch")

# -----------------------------
# ANALYZE
# -----------------------------
if analyze_btn or ai_btn:

    if resume_file is None or job_text.strip() == "":
        st.warning("Please upload resume and enter job description")
        st.stop()

    with st.spinner("Analyzing resume..."):
        resume_text = extract_text_from_file(resume_file)
        if not resume_text:
            st.error("No text found in the resume (is it a scanned image PDF?)")
            st.stop()

        result = ats_score(resume_text, job_text)

        st.session_state["resume_text"] = resume_text
        st.session_state["job_text"] = job_text
        st.session_state["result"] = result
        st.session_state.pop("ai", None)
        st.session_state.pop("new_result", None)

# -----------------------------
# AI FEEDBACK + OPTIMIZATION
# -----------------------------
def upgrade_resume(resume_text, job_text, original, model, style, attempts=3):
    # rewrite, score it, and if it didn't clearly beat the original, try again telling
    # the LLM what it lost / still misses. Keep the best-scoring rewrite.
    best_ai, best_score, retry_note = None, None, ""
    for attempt in range(attempts):
        ai, error = groq_resume_feedback(resume_text, job_text, original["score"], original["missing"],
                                         model, style, original["matched"], retry_note)
        if error:
            return (best_ai, best_score, None) if best_ai else (None, None, error)

        new = ats_score(ai.get("updated_resume", "") or resume_text, job_text)
        if best_score is None or new["score"] > best_score["score"]:
            best_ai, best_score = ai, new
        if best_score["score"] >= original["score"] + 2:
            break

        lost = [k for k in original["matched"] if k not in new["matched"]]
        retry_note = (f"\nPREVIOUS ATTEMPT SCORED {new['score']:.1f}% (original {original['score']:.1f}%) - NOT GOOD ENOUGH.\n"
                      f"It dropped these keywords the original had: {', '.join(lost) or 'none'}.\n"
                      f"Still missing: {', '.join(new['missing']) or 'none'}.\n"
                      "Keep ALL original content and keywords, and work in more of the job's exact terms where truthful.\n")
    return best_ai, best_score, None

if ai_btn and "resume_text" in st.session_state:
    with st.spinner("Getting AI feedback and upgrading resume (it checks the new score and retries if needed)..."):
        ai, new_result, error = upgrade_resume(
            st.session_state["resume_text"],
            st.session_state["job_text"],
            st.session_state["result"],
            groq_model,
            style
        )

    if error:
        st.error(error)
    else:
        st.session_state["ai"] = ai
        st.session_state["updated_resume"] = ai.get("updated_resume", "")
        st.session_state["new_result"] = new_result

# -----------------------------
# RESULTS
# -----------------------------
if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()

    m1, m2 = st.columns(2)
    m1.metric("ATS Score", f"{result['score']:.2f}%")
    m2.metric("Semantic Match", f"{result['semantic']:.1f}%",
              help=f"Projection-layer similarity {result['projection']:.2f}% (rescaled). Raw GAT similarity {result['gat']:.2f}%.")

    if result["score"] < 80:
        st.error("⚠️ Not a good fit for this job.")
    else:
        st.success("✅ Good fit for this job.")

    with st.expander("Profile Summary"):
        st.write(generate_summary(st.session_state["resume_text"]))

if "ai" in st.session_state:
    ai = st.session_state["ai"]
    st.divider()
    st.subheader("🤖 AI Feedback")

    verdict = ai.get("verdict", "")
    msg = f"**{verdict}** - {ai.get('verdict_reason', '')}"
    if "NOT" in verdict.upper():
        st.error(msg)
    elif "PARTIAL" in verdict.upper():
        st.warning(msg)
    else:
        st.success(msg)

    tab1, tab2, tab3 = st.tabs(["✨ Improved Resume", "📝 Feedback", "🔁 Bullet Rewrites"])

    with tab2:
        c1, c2 = st.columns(2)
        c1.markdown("**💪 Strengths**")
        for s in ai.get("strengths", []):
            c1.write(f"- {s}")
        c2.markdown("**🛠 Improvements**")
        for s in ai.get("improvements", []):
            c2.write(f"- {s}")
        st.markdown("**📚 Missing skills (learn these / add only if you have them)**")
        st.write(", ".join(ai.get("missing_skills", [])) or "None")

    with tab3:
        for b in ai.get("rewritten_bullets", []):
            st.markdown(f"~~{b.get('before', '')}~~")
            st.markdown(f"➡️ **{b.get('after', '')}**")
            st.write("")

    with tab1:
        updated_resume = st.text_area("Updated Resume (you can edit it)", key="updated_resume", height=450)
        st.caption("The AI is told not to invent experience - still, review every line before sending.")

        if st.button("↻ Re-score my edits"):
            with st.spinner("Re-evaluating..."):
                st.session_state["new_result"] = ats_score(updated_resume, st.session_state["job_text"])

        # ✅ DOWNLOAD BUTTONS
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 Download PDF", create_pdf(updated_resume), "Improved_Resume.pdf", "application/pdf", width="stretch")
        d2.download_button("📥 Download Word", create_docx(updated_resume), "Improved_Resume.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch")
        d3.download_button("📥 Download TXT", updated_resume, "Improved_Resume.txt", "text/plain", width="stretch")

    if "new_result" in st.session_state:
        new = st.session_state["new_result"]
        old = st.session_state["result"]

        st.subheader("📊 Score Comparison")
        s1, s2 = st.columns(2)
        s1.metric("Original Score", f"{old['score']:.2f}%")
        s2.metric("Improved Score", f"{new['score']:.2f}%", f"{new['score'] - old['score']:+.2f}%")

        if new["score"] > old["score"]:
            st.success("🚀 Resume improved!")
        else:
            st.warning("⚠️ The AI couldn't raise the score for this job after 3 tries. This usually means the job "
                       "asks for skills your resume doesn't show - check the Feedback tab for the missing skills.")
