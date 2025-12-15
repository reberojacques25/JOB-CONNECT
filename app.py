import streamlit as st
import pandas as pd
import numpy as np
import requests
from supabase import create_client
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config("Job Connect Rwanda", layout="wide")

# --------------------------------------------------
# SECRETS
# --------------------------------------------------
ADZUNA_APP_ID = st.secrets["ADZUNA_APP_ID"]
ADZUNA_API_KEY = st.secrets["ADZUNA_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --------------------------------------------------
# CLIENTS
# --------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Students", "Jobs (Adzuna)", "Matching", "AI Interview"]
)

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    st.title("🇷🇼 Job Connect Rwanda — Intelligent Hiring Demo")

    st.markdown("""
    **What this demo proves:**
    - University performance + skills ingestion
    - Live jobs from Adzuna
    - AI-based job matching
    - Automated AI interview scoring
    - Secure database persistence (Supabase)
    """)

# --------------------------------------------------
# STUDENTS
# --------------------------------------------------
elif page == "Students":
    st.title("🎓 Student Data")

    uploaded = st.file_uploader("Upload students CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)

        if st.button("Save to Database"):
            for _, row in df.iterrows():
                supabase.table("students").insert({
                    "name": row["student_name"],
                    "degree": row["degree"],
                    "skills": row["skills"],
                    "marks": {
                        "python": row["python"],
                        "sql": row["sql"],
                        "ml": row["ml"],
                        "communication": row["communication"],
                        "statistics": row["statistics"]
                    }
                }).execute()
            st.success("Students saved to Supabase")

# --------------------------------------------------
# JOBS (ADZUNA)
# --------------------------------------------------
elif page == "Jobs (Adzuna)":
    st.title("💼 Live Jobs (Adzuna)")

    keyword = st.text_input("Job keyword", "data analyst")
    country = st.selectbox("Country", ["rw", "ke", "ug", "tz"])

    if st.button("Fetch Jobs"):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_API_KEY,
            "what": keyword,
            "results_per_page": 10
        }

        data = requests.get(url, params=params).json()["results"]

        jobs = []
        for j in data:
            jobs.append({
                "title": j["title"],
                "company": j["company"]["display_name"],
                "description": j["description"],
                "url": j["redirect_url"]
            })

        jobs_df = pd.DataFrame(jobs)
        st.dataframe(jobs_df)

        if st.button("Save Jobs"):
            for job in jobs:
                supabase.table("jobs").insert(job).execute()
            st.success("Jobs saved to Supabase")

# --------------------------------------------------
# MATCHING
# --------------------------------------------------
elif page == "Matching":
    st.title("🤝 AI Matching Engine")

    students = supabase.table("students").select("*").execute().data
    jobs = supabase.table("jobs").select("*").execute().data

    if not students or not jobs:
        st.warning("Upload students and jobs first.")
    else:
        student_text = [s["skills"] for s in students]
        job_text = [j["description"] for j in jobs]

        tfidf = TfidfVectorizer(stop_words="english")
        vectors = tfidf.fit_transform(student_text + job_text)

        sims = cosine_similarity(
            vectors[:len(student_text)],
            vectors[len(student_text):]
        )

        matches = []
        for i, s in enumerate(students):
            best = np.argmax(sims[i])
            matches.append({
                "Student": s["name"],
                "Job": jobs[best]["title"],
                "Company": jobs[best]["company"],
                "Score": round(sims[i][best], 2)
            })

        st.dataframe(pd.DataFrame(matches))

# --------------------------------------------------
# AI INTERVIEW (GEMINI)
# --------------------------------------------------
elif page == "AI Interview":
    st.title("🎤 AI Interview (Gemini)")

    student = st.text_input("Student Name")
    job = st.text_input("Job Title")

    question = st.selectbox(
        "Interview Question",
        [
            "Tell me about yourself",
            "Describe a project you worked on",
            "Why should we hire you?",
            "Explain a technical challenge you solved"
        ]
    )

    answer = st.text_area("Candidate Answer")

    if st.button("Evaluate"):
        prompt = f"""
        Evaluate this interview answer on clarity, relevance, and technical depth.
        Score from 0 to 100 and give feedback.

        Question: {question}
        Answer: {answer}
        """

        response = model.generate_content(prompt)
        feedback = response.text

        score = int("".join(filter(str.isdigit, feedback))[:2] or 75)

        supabase.table("interviews").insert({
            "student_name": student,
            "job_title": job,
            "score": score,
            "feedback": feedback
        }).execute()

        st.success(f"Interview Score: {score}/100")
        st.markdown(feedback)
