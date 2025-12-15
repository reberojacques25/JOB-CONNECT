# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from supabase import create_client, Client
import time

st.set_page_config(page_title="Job Connect Rwanda Demo", layout="wide")

# -----------------------------
# SECRETS & DATABASE INIT
# -----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ADZUNA_APP_ID = st.secrets["ADZUNA_APP_ID"]
ADZUNA_API_KEY = st.secrets["ADZUNA_API_KEY"]

# -----------------------------
# JOB FETCHING FUNCTION
# -----------------------------
def fetch_jobs_adzuna(keyword="developer", country="rw", max_pages=5, results_per_page=20):
    all_jobs = []
    for page in range(1, max_pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_API_KEY,
            "results_per_page": results_per_page,
            "what": keyword
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if "results" not in data:
                st.warning(f"No results on page {page}. Response: {data}")
                break
            all_jobs.extend(data["results"])
            time.sleep(0.5)  # avoid hitting rate limits
        except Exception as e:
            st.error(f"Error fetching jobs: {e}")
            break
    return all_jobs

# -----------------------------
# DYNAMIC STUDENT INSERT FUNCTION
# -----------------------------
def insert_students_dynamic(df: pd.DataFrame):
    for _, row in df.iterrows():
        # Separate marks from other fields dynamically
        base_cols = ["name", "degree", "skills"]
        marks = {k: float(row[k]) for k in df.columns if k not in base_cols}
        student_data = {col: row[col] for col in base_cols}
        student_data["marks"] = marks

        # Insert into Supabase
        supabase.table("students").insert(student_data).execute()

# -----------------------------
# AI INTERVIEW SCORING (DEMO)
# -----------------------------
def ai_score_interview(answer: str) -> dict:
    # Placeholder for Gemini API call
    # Example:
    # response = requests.post(GEMINI_API_ENDPOINT, headers=headers, json={"text": answer})
    # score = response.json()["score"]
    # feedback = response.json()["feedback"]
    score = np.random.randint(50, 95)
    feedback = ["Clear answer", "Mention technical skills", "Structured response"]
    return {"score": score, "feedback": feedback}

# -----------------------------
# MULTI-PAGE NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Upload Students", "Job Listings", "Matching", "AI Interview Demo"]
)

# -----------------------------
# PAGE: Home
# -----------------------------
if page == "Home":
    st.title("🇷🇼 Job Connect Rwanda — Demo Platform")
    st.markdown("""
    ### Demo Features
    - Upload dynamic student CSVs (different subjects per file)
    - Fetch jobs from Adzuna (multiple pages)
    - AI-powered interview scoring
    - Supabase persistence
    - Multi-page workflow
    """)

# -----------------------------
# PAGE: Upload Students
# -----------------------------
elif page == "Upload Students":
    st.title("📤 Upload Student Data")
    st.markdown("""
    Upload a CSV with at least these columns:
    - **name**
    - **degree**
    - **skills** (comma-separated)
    
    All other columns will be treated as subject marks dynamically.
    """)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)
        if st.button("Save Students to Database"):
            insert_students_dynamic(df)
            st.success("Students saved successfully!")

# -----------------------------
# PAGE: Job Listings
# -----------------------------
elif page == "Job Listings":
    st.title("📄 Job Listings (Adzuna)")
    keyword = st.text_input("Job keyword", "developer")
    country = st.selectbox("Country", ["rw", "ke", "ug", "tz"])
    max_pages = st.slider("Pages to fetch", 1, 10, 3)
    if st.button("Fetch Jobs"):
        jobs = fetch_jobs_adzuna(keyword, country, max_pages)
        if jobs:
            df_jobs = pd.DataFrame(jobs)[["title", "company", "location", "description", "redirect_url"]]
            st.dataframe(df_jobs)
            # Save to Supabase
            for job in jobs:
                supabase.table("jobs").insert({
                    "title": job["title"],
                    "company": job.get("company", {}).get("display_name", ""),
                    "description": job["description"],
                    "url": job["redirect_url"]
                }).execute()
            st.success(f"{len(jobs)} jobs fetched and saved!")

# -----------------------------
# PAGE: Matching
# -----------------------------
elif page == "Matching":
    st.title("🤝 Job Matching Engine")
    # Fetch students & jobs from Supabase
    students = supabase.table("students").select("*").execute().data
    jobs = supabase.table("jobs").select("*").execute().data
    if not students or not jobs:
        st.warning("Upload students and fetch jobs first.")
    else:
        results = []
        for s in students:
            student_skills = [x.strip() for x in s["skills"].split(",")]
            for j in jobs:
                job_skills = j.get("description", "").lower().split()  # simple token matching
                overlap = set(student_skills).intersection(set(job_skills))
                score = round(len(overlap) / max(len(job_skills), 1) * 100, 2)
                results.append({
                    "Student": s["name"],
                    "Degree": s["degree"],
                    "Job": j["title"],
                    "Company": j["company"],
                    "Match Score": score
                })
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
        strong_matches = df_results[df_results["Match Score"] >= 50]
        st.markdown("### Strong Matches (Score ≥ 50%)")
        st.dataframe(strong_matches)

# -----------------------------
# PAGE: AI Interview Demo
# -----------------------------
elif page == "AI Interview Demo":
    st.title("🎤 AI Interview Simulation")
    question = st.selectbox("Choose a question:", [
        "Tell me about yourself.",
        "Why are you interested in this position?",
        "Describe a project where you solved a problem.",
        "What skills make you a strong candidate?"
    ])
    answer = st.text_area("Your answer:")
    if st.button("Analyze Answer"):
        result = ai_score_interview(answer)
        st.success(f"Score: {result['score']} / 100")
        st.markdown("### Feedback:")
        for f in result["feedback"]:
            st.write(f"- {f}")
