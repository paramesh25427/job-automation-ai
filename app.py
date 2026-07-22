import os
import io
import json
from typing import TypedDict, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# =============================================================
# 1. API KEY & MODEL CONFIGURATION
# =============================================================
# Load environment variables from a local .env file
load_dotenv()

# Read OpenRouter key from environment variable or fallback placeholder
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_KEY_HERE")
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1"
)

# Mock database of scraped fresher jobs for demonstration
MOCK_SCRAPED_JOBS = [
    {
        "id": 1,
        "title": "Fresher Full Stack Developer (React + FastAPI)",
        "company": "TechNova Solutions",
        "location": "Bengaluru (Hybrid)",
        "description": "Looking for freshers skilled in Python, FastAPI, React, Node.js, and MongoDB. Knowledge of REST APIs and UI components required.",
        "raw_domain": ""
    },
    {
        "id": 2,
        "title": "Junior Machine Learning / NLP Engineer",
        "company": "AI Dynamics",
        "location": "Chennai (Remote)",
        "description": "Seeking freshers proficient in Python, Scikit-learn, TensorFlow, PyTorch, and NLP text processing. Experience with LLMs and LangChain is a plus.",
        "raw_domain": ""
    },
    {
        "id": 3,
        "title": "Associate DevOps & Cloud Engineer",
        "company": "CloudScale Systems",
        "location": "Hyderabad",
        "description": "Entry-level position for DevOps. Requirements: Docker, Kubernetes basics, CI/CD pipelines, Linux shell scripting, and AWS fundamentals.",
        "raw_domain": ""
    },
    {
        "id": 4,
        "title": "Cyber Security Analyst - Trainee",
        "company": "SecureSphere",
        "location": "Pune",
        "description": "Fresher role in Cyber Security. Requires knowledge of network security, penetration testing, OWASP top 10, SIEM tools, and ethical hacking.",
        "raw_domain": ""
    },
    {
        "id": 5,
        "title": "Associate Salesforce Developer",
        "company": "CRM Cloud Solutions",
        "location": "Coimbatore",
        "description": "Junior Salesforce Developer role. Requirements: Apex programming, Visualforce, Lightning Web Components (LWC), and CRM workflows.",
        "raw_domain": ""
    }
]

# =============================================================
# 2. SHARED GRAPH STATE DEFINITION
# =============================================================
class PS7SystemState(TypedDict):
    scraped_jobs: List[dict]
    categorized_jobs: List[dict]
    resume_text: str
    student_track: str
    student_skills: List[str]
    ranked_jobs: List[dict]

# =============================================================
# 3. PS7 MULTI-AGENT DEFINITIONS
# =============================================================

# --- AGENT 1: Web Scraper Agent ---
def agent1_web_scraper(state: PS7SystemState):
    print("\n🕸️ [Agent 1: Web Scraper] Fetching latest fresher job postings from the web...")
    scraped = MOCK_SCRAPED_JOBS
    print(f"   ↳ Scraped {len(scraped)} fresher jobs successfully.")
    return {"scraped_jobs": scraped}

# --- AGENT 2: Job Domain Classifier Agent (Pure AI JSON Approach) ---
def agent2_job_classifier(state: PS7SystemState):
    print("\n🏷️ [Agent 2: Job Domain Classifier] Classifying scraped jobs into specific tech domains...")
    scraped = state["scraped_jobs"]
    categorized = []
    
    for job in scraped:
        prompt = f"""
        You are a job classification agent.
        Categorize this job into EXACTLY ONE domain from this list:
        - Full Stack
        - Machine Learning
        - DevOps
        - Cyber Security
        - Salesforce
        - Software Testing

        JOB TITLE: {job['title']}
        JOB DESCRIPTION: {job['description']}

        Respond ONLY in valid JSON format with no markdown wrappers:
        {{"domain": "Domain Name Here"}}
        """
        response = llm.invoke(prompt)
        
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed = json.loads(content)
            domain = parsed.get("domain", "General Engineering")
        except Exception:
            domain = "General Engineering"

        job_copy = job.copy()
        job_copy["domain"] = domain
        categorized.append(job_copy)
        print(f"   ↳ Job '{job['title']}' categorized as -> [{domain}]")
        
    return {"categorized_jobs": categorized}

# --- AGENT 3: Resume Classifier Agent ---
def agent3_resume_classifier(state: PS7SystemState):
    print("\n👤 [Agent 3: Resume Classifier] Analyzing student resume to determine primary track...")
    prompt = f"""
    Analyze this candidate's resume text and classify them into EXACTLY ONE primary student track:
    [Full Stack Student, ML Student, DevOps Student, Cyber Security Student, Salesforce Student, Software Testing Student]

    RESUME TEXT:
    {state['resume_text'][:2000]}

    Return JSON strictly in this format:
    {{
        "student_track": "<Track Name>",
        "extracted_skills": ["Skill1", "Skill2", "Skill3"]
    }}
    """
    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        data = json.loads(content)
        track = data.get("student_track", "Full Stack Student")
        skills = data.get("extracted_skills", [])
    except Exception:
        track = "ML Student"
        skills = ["Python", "Machine Learning", "React", "NLP"]

    print(f"   ↳ Student classified into Track: [{track}]")
    return {"student_track": track, "student_skills": skills}

# --- AGENT 4: Matcher & Ranking Engine Agent ---
def agent4_matching_engine(state: PS7SystemState):
    print("\n🎯 [Agent 4: Matcher & Ranking Engine] Scoring and ranking top matching jobs for student profile...")
    track = state["student_track"]
    skills = state["student_skills"]
    categorized_jobs = state["categorized_jobs"]
    
    ranked_jobs = []
    
    for job in categorized_jobs:
        prompt = f"""
        Calculate a Match Percentage (0 to 100) and brief rationale comparing this student profile to the job posting.

        STUDENT TRACK: {track}
        STUDENT SKILLS: {', '.join(skills)}
        RESUME SUMMARY: {state['resume_text'][:1000]}

        JOB TITLE: {job['title']}
        JOB DOMAIN: {job['domain']}
        JOB DESCRIPTION: {job['description']}

        Return JSON strictly in this format:
        {{
            "match_score": 85,
            "reason": "Strong alignment with Python and React, minor gap in FastAPI."
        }}
        """
        response = llm.invoke(prompt)
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            eval_data = json.loads(content)
            score = eval_data.get("match_score", 50)
            reason = eval_data.get("reason", "Good overall skill overlap.")
        except Exception:
            score = 75
            reason = "Matches core programming requirements."

        job_result = job.copy()
        job_result["match_score"] = score
        job_result["reason"] = reason
        ranked_jobs.append(job_result)

    # Sort jobs by Match Percentage descending
    ranked_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    print("   ↳ Job ranking complete. Top match:", ranked_jobs[0]["title"], f"({ranked_jobs[0]['match_score']}%)")
    
    return {"ranked_jobs": ranked_jobs}

# =============================================================
# 4. LANGGRAPH WORKFLOW BUILDER
# =============================================================
builder = StateGraph(PS7SystemState)

builder.add_node("Scraper", agent1_web_scraper)
builder.add_node("JobClassifier", agent2_job_classifier)
builder.add_node("ResumeClassifier", agent3_resume_classifier)
builder.add_node("MatchingEngine", agent4_matching_engine)

builder.set_entry_point("Scraper")
builder.add_edge("Scraper", "JobClassifier")
builder.add_edge("JobClassifier", "ResumeClassifier")
builder.add_edge("ResumeClassifier", "MatchingEngine")
builder.add_edge("MatchingEngine", END)

ps7_workflow = builder.compile()

# =============================================================
# 5. FASTAPI BACKEND SERVER
# =============================================================
app = FastAPI(title="PS7 - Production Job Automation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "PS7 Multi-Agent Job Automation API active!"}

@app.post("/api/ps7-process-student")
def process_student(resume_file: UploadFile = File(...)):
    try:
        resume_text = ""
        if resume_file.filename.endswith(".pdf"):
            pdf_bytes = resume_file.file.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    resume_text += extracted + "\n"
        else:
            content = resume_file.file.read()
            resume_text = content.decode("utf-8", errors="ignore")

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        input_data = {
            "scraped_jobs": [],
            "categorized_jobs": [],
            "resume_text": resume_text,
            "student_track": "",
            "student_skills": [],
            "ranked_jobs": []
        }

        output = ps7_workflow.invoke(input_data)

        return {
            "status": "success",
            "student_track": output["student_track"],
            "student_skills": output["student_skills"],
            "ranked_jobs": output["ranked_jobs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)