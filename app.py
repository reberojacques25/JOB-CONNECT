# app.py
"""
Job Connect Rwanda — Demo (Multi-page Streamlit app)
- Supabase persistence (enable via secrets: SUPABASE_URL, SUPABASE_KEY)
- Gemini AI integration (enable via secrets: GEMINI_API_KEY, GEMINI_ENDPOINT)
- Job aggregation (demo scrapers for JobInRwanda / RwandaJob) - placeholder
- Matching engine (skills overlap 70% + grade 30% by default, configurable)
- Student profile radar chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import math
import json
from supabase import create_client
from typing import List, Dict

st.set_page_config(page_title="Job Connect Rwanda — Demo", layout="wide")

# -------------------------
# Utilities
# -------------------------
def normalize_skills_field(s) -> List[str]:
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return [x.strip() for x in s]
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return parts

def calculate_skill_score(student_skills: List[str], job_skills: List[str]) -> float:
    if not job_skills:
        return 0.0
    overlap = set([s.lower() for s in student_skills]).intersection(set([j.lower() for j in job_skills]))
    score = (len(overlap) / len(job_skills)) * 100
    return round(score, 2)

def calculate_final_score(skill_score: float, grade: float, skill_weight: float = 0.7) -> float:
    grade_norm = float(np.clip(grade if not pd.isna(grade) else 0, 0, 100))
    final = (skill_score * skill_weight) + (grade_norm * (1 - skill_weight))
    return round(final, 2)

# -------------------------
# Supabase helpers
# -------------------------
def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Supabase client init error: {e}")
        return None

def save_students_to_supabase(df: pd.DataFrame):
    sb = get_supabase_client()
    if sb is None:
        st.warning("Supabase not configured in secrets.")
        return
    table = "students_demo"
    for r in df.to_dict(orient="records"):
        # Supabase expects simple types; convert lists to strings if needed
        if "Skills" in r and isinstance(r["Skills"], list):
            r["Skills"] = ", ".join(r["Skills"])
        try:
            sb.table(table).insert(r).execute()
        except Exception as e:
            st.warning(f"Insert error (row): {e}")
    st.success("Attempted to save students to Supabase (check dashboard).")

def load_students_from_supabase(limit=200):
    sb = get_supabase_client()
    if sb is None:
        st.warning("Supabase not configured.")
        return None
    try:
        resp = sb.from_("students_demo").select("*").limit(limit).execute()
        data = resp.data if hasattr(resp, "data") else resp
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.warning(f"Supabase read error: {e}")
        return None

# -------------------------
# Job aggregation (demo)
# -------------------------
def fetch_jobs_jobinrwanda(limit=6):
    jobs = []
    try:
        url = "https://jobinrwanda.com/search?keyword=&location="
        headers = {"User-Agent": "JobConnectRwandaDemo/1.0 (+https://example.org)"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # Generic extraction fallback (site structure changes often)
            cards = soup.select(".job-card") or soup.select(".search-result") or soup.select("article")
            for c in cards[:limit]:
                title_tag = c.select_one("h2") or c.select_one(".title")
                title = title_tag.get_text(strip=True) if title_tag else "Job"
                desc = c.get_text(separator=" ", strip=True)[:500]
                jobs.append({"Job Title": title, "Required Skills": [], "Description": desc, "Source": "JobInRwanda"})
    except Exception as e:
        # Fail gracefully for demo
        st.info("JobInRwanda fetch: demo fallback or network blocked.")
    return pd.DataFrame(jobs)

def fetch_jobs_rwandajob(limit=6):
    jobs = []
    try:
        url = "https://rwandajob.com/jobs/"
        headers = {"User-Agent": "JobConnectRwandaDemo/1.0 (+https://example.org)"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            cards = soup.select(".job") or soup.select(".listing") or soup.select("article")
            for c in cards[:limit]:
                title_tag = c.select_one("h2") or c.select_one(".job-title")
                title = title_tag.get_text(strip=True) if title_tag else "Job"
                desc = c.get_text(separator=" ", strip=True)[:500]
                jobs.append({"Job Title": title, "Required Skills": [], "Description": desc, "Source": "RwandaJob"})
    except Exception as e:
        st.info("RwandaJob fetch: demo fallback or network blocked.")
    return pd.DataFrame(jobs)

def aggregate_jobs(limit_per_source=4):
    frames = []
    frames.append(fetch_jobs_jobinrwanda(limit_per_source))
    frames.append(fetch_jobs_rwandajob(limit_per_source))
    # LinkedIn placeholder - DO NOT SCRAPE LinkedIn; use API/partnerships
    frames.append(pd.DataFrame([{"Job Title":"LinkedIn (API required)","Required Skills":["API Access Required"],"Description":"Use LinkedIn API","Source":"LinkedIn"}]))
    jobs = pd.concat(frames, ignore_index=True, sort=False)
    if jobs.empty:
        demo = [
            {"Job Title":"Data Analyst","Required Skills":["Python","SQL","Data Cleaning","Statistics"],"Description":"Demo job","Source":"Demo"},
            {"Job Title":"Software Developer","Required Skills":["Python","Django","APIs","Git"],"Description":"Demo job","Source":"Demo"},
            {"Job Title":"Project Assistant","Required Skills":["Communication","Excel","Reporting"],"Description":"Demo job","Source":"Demo"},
        ]
        jobs = pd.DataFrame(demo)
    if "Required Skills" not in jobs.columns:
        jobs["Required Skills"] = [[] for _ in range(len(jobs))]
    # Ensure lists where strings exist
    jobs["Required Skills"] = jobs["Required Skills"].apply(lambda x: normalize_skills_field(x) if not isinstance(x, list) else x)
    return jobs

# -------------------------
# Gemini AI integration (interview analysis, job parsing)
# -------------------------
def call_gemini_api(prompt: str, model: str = "gemini-1.5-mini") -> Dict:
    """
    Demo wrapper to call Gemini-like API.
    Expected secrets:
      - GEMINI_API_KEY (required to actually call)
      - GEMINI_ENDPOINT (optional): full endpoint URL for your Gemini deployment (recommended)
    If endpoint or key missing, returns simulated response for demo.
    """
    api_key = st.secrets.get("GEMINI_API_KEY")
    endpoint = st.secrets.get("GEMINI_ENDPOINT")  # e.g. https://your-gemini-endpoint/v1/models/...
    if not api_key or not endpoint:
        # Simulate a response for demo
        simulated_score = int(np.clip(np.random.normal(78, 8), 50, 95))
        return {"score": simulated_score, "feedback": "Simulated Gemini feedback. Configure GEMINI_API_KEY & GEMINI_ENDPOINT in secrets to enable real calls."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": prompt
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        # Attempt to parse JSON
        return resp.json()
    except Exception as e:
        st.warning(f"Gemini API call failed: {e}")
        # fallback simulated
        return {"score": None, "feedback": f"Gemini call failed: {e}"}

def analyze_interview_answer_with_gemini(answer: str, question: str = ""):
    prompt = (
        f"You are a hiring assistant. Given the interview question: '{question}' and the candidate answer: '''{answer}''', "
        "provide a JSON object with fields: score (0-100), strengths (list), weaknesses (list), suggestions (string). "
        "Be concise and specific."
    )
    resp = call_gemini_api(prompt)
    # If resp is simulated (has score int), format accordingly
    if isinstance(resp, dict) and "score" in resp and isinstance(resp["score"], int):
        return resp
    # If the real Gemini returns structured JSON, try to interpret it; else wrap text
    try:
        if isinstance(resp, dict):
            return resp
        else:
            return {"score": None, "feedback_raw": str(resp)}
    except Exception as e:
        return {"score": None, "feedback_raw": f"Parse error: {e}"}

# -------------------------
# Radar chart plotting
# -------------------------
def plot_student_radar(skills: List[str], grades_map: Dict[str,float] = None):
    labels = skills if skills else ["Skill A","Skill B","Skill C","Skill D"]
    n = len(labels)
    if grades_map:
        values = [grades_map.get(lbl, 60) for lbl in labels]
    else:
        values = [int(np.clip(np.random.normal(75,8),40,100)) for _ in labels]
    values += values[:1]
    angles = [i / float(n) * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([20,40,60,80,100])
    ax.set_ylim(0,100)
    st.pyplot(fig)

# -------------------------
# Matching engine
# -------------------------
def run_matching(student_df: pd.DataFrame, jobs_df: pd.DataFrame, skill_weight: float=0.7) -> pd.DataFrame:
    rows = []
    for _, s in student_df.iterrows():
        name = s.get("Name")
        program = s.get("Program", "")
        skills = normalize_skills_field(s.get("Skills",""))
        grade = s.get("Grade", 0)
        for _, j in jobs_df.iterrows():
            job_title = j.get("Job Title", "")
            job_skills = j.get("Required Skills") or []
            skill_score = calculate_skill_score(skills, job_skills)
            final = calculate_final_score(skill_score, float(grade if not pd.isna(grade) else 0), skill_weight)
            rows.append({
                "Student Name": name,
                "Program": program,
                "Skills": ", ".join(skills),
                "Grade": grade,
                "Job Title": job_title,
                "Skill Match (%)": skill_score,
                "Final Match (%)": final,
                "Source": j.get("Source","aggregated")
            })
    return pd.DataFrame(rows)

# -------------------------
# Sample CSVs (fallback / quick load)
# -------------------------
SAMPLE_STUDENTS_CSV = """Name,Program,Skills,Grade
Alice Mukamana,Computer Science,"Python, SQL, Data Analysis, Git",85
Eric Nshimiyimana,Information Systems,"Excel, Communication, Reporting",72
John Mugisha,Software Engineering,"Python, Django, APIs, Git",91
Aline Uwase,Statistics,"Statistics, R, Data Cleaning, Visualization",88
Grace Umuhoza,Computer Science,"HTML, CSS, JavaScript, React",72
Kevin Habumugisha,Data Science,"Python, Machine Learning, SQL, Statistics",94
Sarah Ishimwe,Business IT,"Communication, Problem Solving, Excel",78
"""

SAMPLE_JOBS_CSV = """Job Title,Required Skills,Experience Level
Data Analyst,"Python, SQL, Data Cleaning, Statistics",Entry
Software Developer,"Python, Django, APIs, Git",Junior
Frontend Developer,"HTML, CSS, JavaScript, React",Junior
Project Assistant,"Communication, Excel, Reporting",Entry
Machine Learning Intern,"Python, Machine Learning, Statistics",Entry
IT Support Assistant,"Networking, Troubleshooting, Communication",Entry
"""

# -------------------------
# Streamlit UI - multi page
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home","Upload Students","Jobs","Matching","Profiles","AI Interview","Admin"])

# HOME
if page == "Home":
    st.title("🇷🇼 Job Connect Rwanda — Demo (Supabase + Gemini)")
    st.markdown("""
    Features:
    - Upload student skills + grades (CSV)
    - Aggregate job listings (demo)
    - Matching engine (skills + grades)
    - Gemini-driven AI interview analysis (configure GEMINI_API_KEY & GEMINI_ENDPOINT)
    - Persist students to Supabase (configure SUPABASE_URL & SUPABASE_KEY)
    """)
    st.info("Secrets expected in Streamlit: SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, GEMINI_ENDPOINT (endpoint optional).")

# UPLOAD STUDENTS
elif page == "Upload Students":
    st.header("Upload Student CSV")
    st.markdown("CSV columns: Name, Program, Skills (comma-separated), Grade (0-100)")
    uploaded = st.file_uploader("Upload students.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        # normalize Skills col into comma-joined string if list-like
        if "Skills" in df.columns and df["Skills"].apply(lambda x: isinstance(x, list)).any():
            df["Skills"] = df["Skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        st.session_state["students"] = df
        st.success("Uploaded students.csv")
        st.dataframe(df)
    else:
        st.info("No file uploaded — load sample below to try the demo.")
        if st.button("Load sample students"):
            df = pd.read_csv(pd.compat.StringIO(SAMPLE_STUDENTS_CSV))
            st.session_state["students"] = df
            st.success("Sample students loaded")
            st.dataframe(df)

# JOBS
elif page == "Jobs":
    st.header("Aggregate Jobs (Demo)")
    st.markdown("This aggregates a few public sources (demo). For production replace with robust pipelines and official APIs.")
    jobs = aggregate_jobs(limit_per_source=4)
    st.session_state["jobs"] = jobs
    st.dataframe(jobs[["Job Title","Required Skills","Source"]].head(50))

# MATCHING
elif page == "Matching":
    st.header("Matching Engine")
    if "students" not in st.session_state:
        st.warning("Upload students first (Upload Students tab).")
    else:
        students = st.session_state["students"]
        jobs = st.session_state.get("jobs") or aggregate_jobs(limit_per_source=4)
        if "Required Skills" in jobs.columns:
            jobs["Required Skills"] = jobs["Required Skills"].apply(lambda x: normalize_skills_field(x) if not isinstance(x, list) else x)
        skill_weight = st.slider("Skill weight (0–1): how much skills count vs grade", 0.0, 1.0, 0.7, 0.05)
        results = run_matching(students, jobs, skill_weight=skill_weight)
        st.session_state["matches"] = results
        st.success("Matching completed")
        st.dataframe(results.sort_values("Final Match (%)", ascending=False).head(200))
        st.markdown("### Strong matches (Final Match ≥ 70%)")
        st.dataframe(results[results["Final Match (%)"] >= 70].sort_values("Final Match (%)", ascending=False))

# PROFILES
elif page == "Profiles":
    st.header("Student Profile Visualization")
    if "students" not in st.session_state:
        st.warning("Upload students first.")
    else:
        students = st.session_state["students"]
        names = students["Name"].tolist()
        sel = st.selectbox("Select student", names)
        s = students[students["Name"] == sel].iloc[0]
        st.subheader(f"{s['Name']} — {s.get('Program','')}")
        skills = normalize_skills_field(s.get("Skills",""))
        grade = s.get("Grade", 70)
        st.write("**Skills:**", ", ".join(skills))
        st.write("**Grade:**", grade)
        st.markdown("### Radar chart (demo)")
        grades_map = {sk: float(np.clip(grade + np.random.randint(-6,7), 40, 100)) for sk in skills} if skills else None
        plot_student_radar(skills if skills else ["A","B","C","D"], grades_map)

# AI INTERVIEW
elif page == "AI Interview":
    st.header("AI Interview Analysis (Gemini)")
    st.markdown("Enter an answer and get feedback from Gemini (requires GEMINI_API_KEY and GEMINI_ENDPOINT in secrets for real results).")
    question = st.selectbox("Question:", [
        "Tell me about yourself.",
        "Why are you interested in this position?",
        "Describe a project where you solved a problem.",
        "What skills make you a strong candidate?"
    ])
    answer = st.text_area("Paste candidate's answer:")
    if st.button("Analyze answer with Gemini"):
        with st.spinner("Calling Gemini..."):
            result = analyze_interview_answer_with_gemini(answer, question)
            st.write(result)
            if result.get("score") is not None:
                st.success(f"Score: {result['score']}/100")

# ADMIN
elif page == "Admin":
    st.header("Admin & Persistence")
    st.markdown("Persist or load session data to/from Supabase. Configure secrets: SUPABASE_URL and SUPABASE_KEY.")
    supabase_ready = bool(st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY"))
    st.write("Supabase configured:", supabase_ready)
    if st.button("Show session keys"):
        st.write({"students": "present" if "students" in st.session_state else "no",
                  "jobs": "present" if "jobs" in st.session_state else "no",
                  "matches": "present" if "matches" in st.session_state else "no"})
    if supabase_ready:
        if st.button("Save students to Supabase"):
            df = st.session_state.get("students")
            if df is None:
                st.warning("No students to save.")
            else:
                save_students_to_supabase(df)
        if st.button("Load students from Supabase"):
            df = load_students_from_supabase()
            if df is not None:
                st.session_state["students"] = df
                st.success("Loaded students from Supabase (if any).")
                st.dataframe(df)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Demo — respect platform TOS. Configure keys in Streamlit secrets for Supabase and Gemini.")
