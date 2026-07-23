import io
import json
import os
import time
import urllib.parse
from datetime import datetime
from typing import List, TypedDict

import bcrypt
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from pypdf import PdfReader
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

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
    pool_pre_ping=True,
    pool_recycle=280,
    pool_size=5,
    max_overflow=10,
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

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
    temperature=0.2
)

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

# =============================================================
# 3. LANGGRAPH MULTI-AGENT PIPELINE
# =============================================================
class PS7SystemState(TypedDict):
    resume_text: str
    student_track: str
    student_skills: List[str]
    scraped_jobs: List[dict]
    categorized_jobs: List[dict]
    ranked_jobs: List[dict]

# AGENT 1: RESUME CLASSIFIER & SKILL EXTRACTION
def agent1_resume_classifier(state: PS7SystemState):
    print("\n👤 [Agent 1: Resume Classifier] Analyzing student resume track & skills...")
    prompt = f"""
    Classify this resume into EXACTLY ONE student track:
    [Full Stack Student, ML Student, DevOps Student, Cyber Security Student, Salesforce Student]

    RESUME TEXT:
    {state['resume_text'][:2000]}

    Return JSON strictly in this format:
    {{
        "student_track": "Track Name",
        "extracted_skills": ["Skill1", "Skill2", "Skill3"]
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
        track = data.get("student_track", "Software Developer")
        skills = data.get("extracted_skills", [])
    except Exception:
        track = "Software Developer"
        skills = ["Python", "JavaScript"]

    print(f"   ↳ Student track: [{track}] | Skills extracted: {skills}")
    return {"student_track": track, "student_skills": skills}

# DYNAMIC FILTER: Rejects non-job page titles
def is_valid_job_title(title_text: str) -> bool:
    t = title_text.strip().lower()
    # Reject generic non-job portal/landing pages
    generic_words = ["careers", "home", "jobs", "all jobs", "search results", "workday", "job openings"]
    if t in generic_words or len(t) < 4:
        return False
    return True

# AGENT 2: TARGETED DIRECT JOB SCRAPER
def agent2_web_scraper(state: PS7SystemState):
    raw_track = state["student_track"]
    skills = state["student_skills"]
    
    clean_track = raw_track.replace("Student", "").strip()
    top_skill = skills[0] if skills else "Python"
    
    queries = [
        f'"{top_skill}" "Fresher" hiring India site:lever.co OR site:greenhouse.io OR site:workday.com',
        f'"{clean_track}" "Fresher" "Apply" site:linkedin.com/jobs/view/',
        f'fresher "{top_skill} developer" job opening India site:naukri.com/job-listings-'
    ]
    
    live_jobs = []

    # Strategy A: DDGS
    try:
        for q in queries:
            if len(live_jobs) >= 5:
                break
            print(f"\n🕸️ [Agent 2: Live Web Scraper] Querying web: '{q}'...")
            results = list(DDGS().text(q, max_results=5))
            for res in results:
                href = res.get("href", "")
                title = res.get("title", "")
                if href and href.startswith("http") and is_valid_job_title(title) and not any(j["url"] == href for j in live_jobs):
                    live_jobs.append({
                        "id": len(live_jobs) + 1,
                        "raw_title": title,
                        "raw_snippet": res.get("body", "Live posting found on web.")[:500],
                        "url": href
                    })
        if live_jobs:
            print(f"   ↳ Retrieved {len(live_jobs)} raw live listings via DDGS!")
            return {"scraped_jobs": live_jobs}
    except Exception as e:
        print(f"   ⚠️ DDGS notice: {e}. Switching to direct HTML web parser...")

    # Strategy B: Fallback HTML Scraper
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    for q in queries:
        if len(live_jobs) >= 5:
            break
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
            resp = requests.get(search_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                results = soup.find_all("div", class_="result")
                for res in results:
                    title_elem = res.find("a", class_="result__a")
                    snippet_elem = res.find("a", class_="result__snippet")
                    if title_elem and title_elem.get("href"):
                        href = title_elem.get("href")
                        title_text = title_elem.get_text(strip=True)
                        if "uddg=" in href:
                            href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
                        if href.startswith("http") and is_valid_job_title(title_text) and not any(j["url"] == href for j in live_jobs):
                            live_jobs.append({
                                "id": len(live_jobs) + 1,
                                "raw_title": title_text,
                                "raw_snippet": snippet_elem.get_text(strip=True) if snippet_elem else "Live job posting found on web.",
                                "url": href
                            })
        except Exception as ex:
            print(f"   ⚠️ HTML Scraper error: {ex}")

    print(f"   ↳ Total raw live listings found: {len(live_jobs)}")
    return {"scraped_jobs": live_jobs}

# AGENT 3: DYNAMIC ENTITY EXTRACTION & ACCURATE DOMAIN CLASSIFIER
def agent3_job_classifier(state: PS7SystemState):
    print("\n🏷️ [Agent 3: Entity Extractor & Classifier] Extracting Company Names & Job Titles via AI...")
    scraped = state["scraped_jobs"]
    categorized = []
    
    for job in scraped:
        prompt = f"""
        Analyze this raw web search listing and extract/clean the details:

        RAW TITLE: {job['raw_title']}
        RAW SNIPPET: {job['raw_snippet']}
        SOURCE URL: {job['url']}

        Instructions:
        1. "company": Extract the hiring company name. If not found in text, extract domain brand from URL.
        2. "clean_title": Clean up numbers/noise. MUST BE A VALID SPECIFIC JOB TITLE (e.g., "Software Engineer - Python"). If the raw title is a generic non-job word like "Careers", "Jobs", or "Workday", set "is_generic_page": true.
        3. "domain": Categorize strictly from [Full Stack, Machine Learning, DevOps, Cyber Security, Salesforce, Software Testing].
           IMPORTANT DOMAIN RULE: Prioritize the main role in the TITLE. If title says "Software Engineer", "Python Engineer", "Developer", tag as "Full Stack" or "Machine Learning". DO NOT tag as "Software Testing" unless "QA", "Tester", or "Testing" is explicitly written in the TITLE.

        Return JSON strictly:
        {{
            "company": "Company Name",
            "clean_title": "Clean Job Title",
            "domain": "Domain Name",
            "is_generic_page": false
        }}
        """
        response = safe_llm_invoke(prompt)
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            
            # Skip if AI detected it's a generic portal page
            if parsed.get("is_generic_page") is True or not is_valid_job_title(parsed.get("clean_title", "")):
                print(f"   ⚠️ Filtered out generic portal page: '{job['raw_title']}'")
                continue

            company = parsed.get("company", "Tech Hiring Partner")
            clean_title = parsed.get("clean_title", job["raw_title"])
            domain = parsed.get("domain", "General Engineering")
        except Exception:
            if not is_valid_job_title(job["raw_title"]):
                continue
            company = "Tech Hiring Partner"
            clean_title = job["raw_title"]
            domain = "General Engineering"

        categorized.append({
            "id": job["id"],
            "title": clean_title,
            "company": company,
            "location": "India (Remote/Onsite)",
            "description": job["raw_snippet"],
            "url": job["url"],
            "domain": domain
        })
        print(f"   ↳ Extracted -> Company: [{company}] | Title: [{clean_title[:35]}...] | Domain: [{domain}]")
        
    return {"categorized_jobs": categorized}

# AGENT 4: MATCHING ENGINE & RANKER
def agent4_matching_engine(state: PS7SystemState):
    print("\n🎯 [Agent 4: Matcher Engine] Scoring candidate against scraped jobs...")
    track = state["student_track"]
    skills = state["student_skills"]
    categorized_jobs = state["categorized_jobs"]
    ranked_jobs = []
    
    for job in categorized_jobs:
        prompt = f"""
        Score match percentage (0 to 100) between student resume skills and this job posting.
        STUDENT TRACK: {track} | SKILLS: {', '.join(skills)}
        JOB TITLE: {job['title']} | COMPANY: {job['company']} | DESCRIPTION: {job['description']}
        
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
            reason = eval_data.get("reason", "Matches profile requirements.")
        except Exception:
            score = 60
            reason = "Matches general track skills."

        job_result = job.copy()
        job_result["match_score"] = score
        job_result["reason"] = reason
        ranked_jobs.append(job_result)

    ranked_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    print(f"   ↳ Ranked {len(ranked_jobs)} matched live jobs.")
    return {"ranked_jobs": ranked_jobs}

# Build LangGraph workflow
builder = StateGraph(PS7SystemState)
builder.add_node("ResumeClassifier", agent1_resume_classifier)
builder.add_node("WebScraper", agent2_web_scraper)
builder.add_node("JobClassifier", agent3_job_classifier)
builder.add_node("MatchingEngine", agent4_matching_engine)

builder.set_entry_point("ResumeClassifier")
builder.add_edge("ResumeClassifier", "WebScraper")
builder.add_edge("WebScraper", "JobClassifier")
builder.add_edge("JobClassifier", "MatchingEngine")
builder.add_edge("MatchingEngine", END)

ps7_workflow = builder.compile()

# =============================================================
# 4. FASTAPI SERVER & ENDPOINTS
# =============================================================
app = FastAPI(title="Job Automation AI - Production Engine")

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
    return {"message": "Job Automation API with Live Web Search Engine online!"}

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

        if not resume_text.strip() or len(resume_text.strip()) < 20:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from file. Please ensure you upload a standard, text-selectable PDF or TXT document."
            )

        input_data = {
            "resume_text": resume_text,
            "student_track": "",
            "student_skills": [],
            "scraped_jobs": [],
            "categorized_jobs": [],
            "ranked_jobs": []
        }

        output = ps7_workflow.invoke(input_data)
        
        if not output.get("ranked_jobs"):
            raise HTTPException(
                status_code=404, 
                detail="No live job postings could be retrieved from search engines. Please check your internet connection and try again."
            )

        top_match = output["ranked_jobs"][0]

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)