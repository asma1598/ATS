import streamlit as st

st.set_page_config(page_title="ATS Resume Optimizer with LLM", page_icon="🎯", layout="wide")

# -----------------------------
# PAGES
#   views/home.py      - landing page
#   views/employer.py  - bulk resume screening for companies
#   views/candidate.py - resume score + AI upgrade for job seekers
#   ats_core.py        - shared models, scoring, LLM and export code
# -----------------------------
pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/employer.py", title="Employer", icon="🏢"),
    st.Page("views/candidate.py", title="Candidate", icon="👤"),
]

st.navigation(pages).run()
