#  JobAutomation.ai

An autonomous, multi-agent recruitment platform powered by **LangGraph**, **Ollama (Llama 3.2)**, **FastAPI**, and **Neon PostgreSQL**, wrapped in a sleek, high-performance dark-mode UI styled with Tailwind CSS.

---

##  Key Features

* **Multi-Agent AI Pipeline (LangGraph):**
  1. **Scraper Agent:** Fetches fresher job openings.
  2. **Job Classifier Agent:** Categorizes job domains using local LLM inference.
  3. **Resume Classifier Agent:** Extracts candidate tracking data and key skills from uploaded PDF/TXT resumes.
  4. **Matching Engine Agent:** Computes precise match percentages and provides contextual AI reasoning.
* **Secure Authentication:** User signup and login powered by `bcrypt` password hashing and session management.
* **Cloud Persistence:** Live activity logging and database storage using **Neon Serverless PostgreSQL**.
* **Modern Interface:** Self-contained single-page dashboard featuring modern typography, blur effects, glassmorphism, and responsive feedback states.

---

##  Tech Stack

* **Backend:** Python, FastAPI, SQLAlchemy, LangGraph, LangChain, PyPDF, Passlib
* **Database:** Neon PostgreSQL (Cloud)
* **AI Model:** Local LLM via Ollama (`llama3.2`)
* **Frontend:** HTML5, Tailwind CSS (via CDN), Vanilla JavaScript

---

##  Setup & Installation Instructions

### 1. Clone or Open the Project Directory
Make sure your project files (`app.py` and `index.html`) are in the same folder along with `requirements.txt`.

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt