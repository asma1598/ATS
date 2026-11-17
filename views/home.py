import streamlit as st

st.title("🎯 ATS Resume Optimizer with LLM")
st.markdown("##### One platform for both sides of hiring - choose how you want to use it:")
st.write("")

c1, c2 = st.columns(2, gap="large")

with c1:
    with st.container(border=True):
        st.subheader("🏢 For Employers")
        st.markdown("""
- Upload **hundreds of resumes at once** (PDF, DOCX, TXT or a ZIP)
- Every resume is scored against your **job description**
- Set a **minimum ATS score** and **must-have skills**
- Get a **ranked shortlist** of eligible candidates - the rest are filtered out
- Download the shortlist as **CSV** and the eligible resumes as a **ZIP**
""")
        st.page_link("views/employer.py", label="Screen resumes", icon="➡️")

with c2:
    with st.container(border=True):
        st.subheader("👤 For Candidates")
        st.markdown("""
- Upload **your resume** and paste the **job description** you're applying for
- See your **ATS score** and whether you're a good fit
- Let the AI **redesign your resume** for that job, without inventing experience
- Compare your **old vs new score**
- Download the improved resume as **PDF, Word or TXT**
""")
        st.page_link("views/candidate.py", label="Upgrade my resume", icon="➡️")

st.write("")
st.caption("Scoring: DistilBERT + GAT model · Resume rewriting: Groq LLM · Runs fully on CPU")
