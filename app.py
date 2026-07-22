import io
import json
import os
import time
from datetime import datetime
from typing import List, TypedDict

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from PIL import Image
from pydantic import BaseModel
from pypdf import PdfReader
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# OCR imports
from pdf2image import convert_from_bytes
import pytesseract

# OPTIONAL FOR WINDOWS: If Tesseract is not in your System PATH, un-comment and set this line:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 1. Load environment variables
load_dotenv()

# =============================================================
# 1. DATABASE & AUTH CONFIGURATION (Neon PostgreSQL)
# =============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("CRITICAL ERROR: DATABASE_URL environment variable is missing in .env!")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,   # Verifies connection health before executing queries
    pool_recycle=280,    # Refreshes connections every ~4.5 minutes
    pool_size=5,         # Maintains 5 idle connections
    max_overflow=10,     # Allows temporary bursts up to 15 connections
    connect_args={
        "connect_timeout": 10,  # Allows 10s for Neon DB auto-wake
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# Python 3.14 Compatible Native Bcrypt Functions
def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    activities = relationship("UserActivityDB", back_populates="user")

class UserActivityDB(Base):
    __tablename__ = "user_activities"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    student_track = Column(String)
    top_job_title = Column(String)
    match_score = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("UserDB", back_populates="activities")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# =============================================================
# 2. GROQ LLM WITH RATE-LIMIT SAFEGUARD
# =============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY environment variable is missing in .env!")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# Wrapper function to retry on Groq rate limits (429) automatically
def safe_llm_invoke(prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "rate" in err_msg or "limit" in err_msg) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"⚠️ Groq rate limit hit. Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                raise e
    return llm.invoke(prompt)

MOCK_SCRAPED_JOBS = [
    {
        "id": 1,
        "title": "Fresher Full Stack Developer (React + FastAPI)",
        "company": "TechNova Solutions",
        "location": "Bengaluru (Hybrid)",
        "description": "Looking for freshers skilled in Python, FastAPI, React, Node.js, and MongoDB. Knowledge of REST APIs and UI components required."
    },
    {
        "id": 2,
        "title": "Junior Machine Learning / NLP Engineer",
        "company": "AI Dynamics",
        "location": "Chennai (Remote)",
        "description": "Seeking freshers proficient in Python, Scikit-learn, TensorFlow, PyTorch, and NLP text processing. Experience with LLMs and LangChain is a plus."
    },
    {
        "id": 3,
        "title": "Associate DevOps & Cloud Engineer",
        "company": "CloudScale Systems",
        "location": "Hyderabad",
        "description": "Entry-level position for DevOps. Requirements: Docker, Kubernetes basics, CI/CD pipelines, Linux shell scripting, and AWS fundamentals."
    },
    {
        "id": 4,
        "title": "Cyber Security Analyst - Trainee",
        "company": "SecureSphere",
        "location": "Pune",
        "description": "Fresher role in Cyber Security. Requires knowledge of network security, penetration testing, OWASP top 10, SIEM tools, and ethical hacking."
    },
    {
        "id": 5,
        "title": "Associate Salesforce Developer",
        "company": "CRM Cloud Solutions",
        "location": "Coimbatore",
        "description": "Junior Salesforce Developer role. Requirements: Apex programming, Visualforce, Lightning Web Components (LWC), and CRM workflows."
    }
]

# =============================================================
# 3. HELPER: SMART PDF & OCR TEXT EXTRACTION
# =============================================================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    resume_text = ""
    
    # Step 1: Digital text extraction attempt via pypdf
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + "\n"
    except Exception as e:
        print(f"pypdf extraction notice: {e}")

    # Step 2: Fallback to OCR if pypdf returns empty or minimal text (Photo/Scanned PDF)
    if len(resume_text.strip()) < 20:
        print("📸 Photo/Scanned PDF detected! Running Tesseract OCR fallback...")
        try:
            images = convert_from_bytes(pdf_bytes)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            if ocr_text.strip():
                resume_text = ocr_text
        except Exception as ocr_err:
            print(f"⚠️ OCR extraction failed: {ocr_err}")

    return resume_text

# =============================================================
# 4. LANGGRAPH MULTI-AGENT PIPELINE
# =============================================================
class PS7SystemState(TypedDict):
    scraped_jobs: List[dict]
    categorized_jobs: List[dict]
    resume_text: str
    student_track: str
    student_skills: List[str]
    ranked_jobs: List[dict]

def agent1_web_scraper(state: PS7SystemState):
    print("\n🕸️ [Agent 1: Web Scraper] Fetching latest job postings...")
    return {"scraped_jobs": MOCK_SCRAPED_JOBS}

def agent2_job_classifier(state: PS7SystemState):
    print("\n🏷️ [Agent 2: Job Domain Classifier] Classifying job domains...")
    scraped = state["scraped_jobs"]
    categorized = []
    
    for job in scraped:
        prompt = f"""
        Categorize this job into EXACTLY ONE domain from [Full Stack, Machine Learning, DevOps, Cyber Security, Salesforce, Software Testing]:
        TITLE: {job['title']}
        DESCRIPTION: {job['description']}
        Return ONLY valid JSON: {{"domain": "Domain Name"}}
        """
        response = safe_llm_invoke(prompt)
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
        print(f"   ↳ Job '{job['title']}' -> [{domain}]")
        
    return {"categorized_jobs": categorized}

def agent3_resume_classifier(state: PS7SystemState):
    print("\n👤 [Agent 3: Resume Classifier] Analyzing student resume track...")
    prompt = f"""
    Classify this resume into EXACTLY ONE student track:
    [Full Stack Student, ML Student, DevOps Student, Cyber Security Student, Salesforce Student]

    RESUME TEXT:
    {state['resume_text'][:2000]}

    Return JSON strictly in this format:
    {{
        "student_track": "Track Name",
        "extracted_skills": ["Skill1", "Skill2"]
    }}
    """
    response = safe_llm_invoke(prompt)
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        track = data.get("student_track", "Full Stack Student")
        skills = data.get("extracted_skills", [])
    except Exception:
        track = "ML Student"
        skills = ["Python", "Machine Learning"]

    print(f"   ↳ Student classified as: [{track}]")
    return {"student_track": track, "student_skills": skills}

def agent4_matching_engine(state: PS7SystemState):
    print("\n🎯 [Agent 4: Matcher Engine] Scoring and ranking jobs...")
    track = state["student_track"]
    skills = state["student_skills"]
    categorized_jobs = state["categorized_jobs"]
    ranked_jobs = []
    
    for job in categorized_jobs:
        prompt = f"""
        Score match percentage (0 to 100) between student and job.
        STUDENT TRACK: {track} | SKILLS: {', '.join(skills)}
        JOB TITLE: {job['title']} | DESCRIPTION: {job['description']}
        Return JSON strictly: {{"match_score": 85, "reason": "Reason here"}}
        """
        response = safe_llm_invoke(prompt)
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            eval_data = json.loads(content)
            score = eval_data.get("match_score", 50)
            reason = eval_data.get("reason", "Good match.")
        except Exception:
            score = 75
            reason = "Matches standard requirements."

        job_result = job.copy()
        job_result["match_score"] = score
        job_result["reason"] = reason
        ranked_jobs.append(job_result)

    ranked_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    filtered_jobs = [job for job in ranked_jobs if job["match_score"] > 0]

    if not filtered_jobs and ranked_jobs:
        filtered_jobs = [ranked_jobs[0]]

    print(f"   ↳ Filtered down to {len(filtered_jobs)} relevant matching jobs.")
    return {"ranked_jobs": filtered_jobs}

# Build LangGraph workflow
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
# 5. FASTAPI SERVER & ENDPOINTS
# =============================================================
app = FastAPI(title="Job Automation AI - Production Cloud Engine")

# Production-safe CORS Configuration
frontend_url = os.getenv("FRONTEND_URL", "*")
allowed_origins = [frontend_url] if frontend_url != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if frontend_url != "*" else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Job Automation API with Groq Cloud & Neon Database online!"}

# Signup Endpoint
@app.post("/api/signup")
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = hash_password(user.password)
    new_user = UserDB(email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"success": True, "message": "Registered successfully", "user_id": new_user.id}

# Login Endpoint
@app.post("/api/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid email or password"
        )
    
    return {"success": True, "message": "Login successful", "user_id": db_user.id, "email": db_user.email}

# User Activity History Endpoint
@app.get("/api/history/{user_id}")
def get_user_history(user_id: int, db: Session = Depends(get_db)):
    activities = db.query(UserActivityDB).filter(UserActivityDB.user_id == user_id).all()
    history_list = []
    for act in activities:
        history_list.append({
            "id": act.id,
            "student_track": act.student_track,
            "top_job_title": act.top_job_title,
            "match_score": act.match_score,
            "timestamp": act.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })
    return {"success": True, "history": history_list}

# Dual Routing for Resume Processing
@app.post("/api/ps7-process-student")
@app.post("/api/ps7-process-student/{user_id}")
def process_student_with_db(
    user_id: int = 1, 
    resume_file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not db_user:
        if user_id == 1:
            db_user = UserDB(email="guest@example.com", hashed_password=hash_password("guest123"))
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
        else:
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found.")

    try:
        if resume_file.filename.endswith(".pdf"):
            pdf_bytes = resume_file.file.read()
            resume_text = extract_text_from_pdf(pdf_bytes)
        else:
            content = resume_file.file.read()
            resume_text = content.decode("utf-8", errors="ignore")

        if not resume_text.strip() or len(resume_text.strip()) < 20:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from file. Please ensure the uploaded file contains readable text or clear scanned image quality."
            )

        input_data = {
            "scraped_jobs": [],
            "categorized_jobs": [],
            "resume_text": resume_text,
            "student_track": "",
            "student_skills": [],
            "ranked_jobs": []
        }

        output = ps7_workflow.invoke(input_data)
        top_match = output["ranked_jobs"][0] if output["ranked_jobs"] else {"title": "General Role", "match_score": 70}

        new_activity = UserActivityDB(
            user_id=db_user.id,
            student_track=output["student_track"],
            top_job_title=top_match["title"],
            match_score=top_match["match_score"]
        )
        db.add(new_activity)
        db.commit()

        return {
            "status": "success",
            "student_track": output["student_track"],
            "student_skills": output["student_skills"],
            "ranked_jobs": output["ranked_jobs"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Debug Endpoint
@app.get("/api/debug/data")
def debug_view_data(db: Session = Depends(get_db)):
    users = db.query(UserDB).all()
    activities = db.query(UserActivityDB).all()
    return {
        "users": [{"id": u.id, "email": u.email, "created_at": u.created_at} for u in users],
        "activities": [
            {
                "id": a.id, 
                "user_id": a.user_id, 
                "student_track": a.student_track, 
                "top_job_title": a.top_job_title, 
                "match_score": a.match_score,
                "timestamp": a.timestamp
            } for a in activities
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)