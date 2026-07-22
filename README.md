# 🎓 Multi-Agent Job Automation System (PS7)

An autonomous multi-agent platform designed to streamline fresher job hunting. Using **FastAPI**, **LangGraph**, and **LLMs**, the system automatically scrapes job postings, classifies domains, categorizes student resumes, and ranks top matching job opportunities.

---

## 🤖 Multi-Agent Architecture

The system executes 4 specialized agents sequentially:

1. **🕸️ Agent 1: Web Scraper Agent** – Extracts real-time/fresher job listings.
2. **🏷️ Agent 2: Job Domain Classifier Agent** – Tags postings into core domains (*Full Stack, ML, DevOps, Cyber Security, etc.*).
3. **👤 Agent 3: Resume Classifier Agent** – Analyzes uploaded PDF resumes to identify student tracks and technical skills.
4. **🎯 Agent 4: Student Matcher & Ranking Engine** – Calculates match scores and ranks jobs tailored to the student's profile.

---

## 🚀 Quickstart Guide for Judges / Evaluators

### 1. Clone Repository
```bash
git clone <YOUR-GITHUB-REPO-LINK>
cd project