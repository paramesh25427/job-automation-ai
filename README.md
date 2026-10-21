<div align="center">

<img src="JobAutomation.ai%20Recruitment%20Workflow.png" alt="JobAutomation.ai recruitment workflow" width="100%"/>

### Find opportunities. Understand your fit. Apply smarter.

</div>

---
---

## 🎯 Overview

**JobAutomation.ai** is an AI-powered recruitment platform built to help freshers discover relevant job openings and understand how well their resumes match job requirements.

The application combines a multi-agent workflow with a web dashboard, resume parsing, job classification, and candidate-job matching. It uses a local language model through Ollama and cloud-based PostgreSQL persistence through Neon.

## ✨ Key Features

### 🤖 Multi-Agent AI Pipeline

The workflow uses LangGraph to coordinate specialized agents:

| Agent                      | Responsibility                                                              |
| -------------------------- | --------------------------------------------------------------------------- |
| 🔎 Scraper Agent           | Fetches fresher job openings from configured sources.                       |
| 🧭 Job Classifier Agent    | Classifies job openings into relevant domains using local LLM inference.    |
| 📄 Resume Classifier Agent | Extracts skills and candidate information from uploaded PDF or TXT resumes. |
| 🎯 Matching Engine Agent   | Calculates a job-match percentage and generates contextual AI reasoning.    |

### 🔐 Authentication

- User registration and login.
- Password hashing with `bcrypt`.
- Session management for authenticated users.

### ☁️ Persistent Storage

- Neon Serverless PostgreSQL for cloud database storage.
- SQLAlchemy for database interaction.
- Live activity logging and persistence.

### 🎨 Modern User Interface

- Dark-mode dashboard.
- Tailwind CSS styling.
- Glassmorphism and blur effects.
- Responsive layouts and user feedback states.

---

## 🔄 AI Agent Pipeline

\<div align="center">

\<img src="[https://capsule-render.vercel.app/api?type=rect&height=3&color=0:7C3AED,100:4F46E5](https://capsule-render.vercel.app/api?type=rect\&height=3\&color=0:7C3AED,100:4F46E5)" width="100%" alt="Purple divider"/>

**Job Discovery → Job Classification → Resume Analysis → Candidate-Job Matching**

\<sub>LangGraph coordinates the workflow; Ollama provides local LLM inference.\</sub>

\</div>

1. **Discover:** The scraper agent collects job openings from configured sources.
2. **Classify:** The job classifier identifies job domains and categories.
3. **Analyze:** The resume classifier extracts relevant candidate information from uploaded documents.
4. **Match:** The matching engine evaluates candidate-job alignment and provides AI-generated reasoning.
5. **Persist:** Application data and activity records are stored in the configured PostgreSQL database.

---

## 🧰 Tech Stack

| Layer               | Technologies                                |
| ------------------- | ------------------------------------------- |
| Language            | Python                                      |
| API framework       | FastAPI                                     |
| Agent orchestration | LangGraph                                   |
| LLM integration     | Ollama — Llama 3.2                          |
| LLM framework       | LangChain                                   |
| Database            | Neon PostgreSQL                             |
| ORM                 | SQLAlchemy                                  |
| Resume processing   | PyPDF                                       |
| Authentication      | bcrypt / Passlib                            |
| Frontend            | HTML5, Tailwind CSS, Vanilla JavaScript     |
| Deployment support  | Python virtual environments; cloud database |

---

## 📁 Project Structure

The repository currently includes these files:

```text
job-automation-ai/
├── app.py
├── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

- `app.py` — Python application and backend entry point.
- `index.html` — Web dashboard and frontend.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Files and directories excluded from Git tracking.
- `README.md` — Project documentation.

---

## ⚙️ Setup & Installation

### Prerequisites

Install or configure the following before running the application:

- Python
- Ollama with the `llama3.2` model available locally
- A Neon PostgreSQL database
- Git

### 1. Clone the repository

```bash
git clone https://github.com/paramesh25427/job-automation-ai.git
cd job-automation-ai
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the database and environment

Configure your Neon PostgreSQL connection and any required application settings.

Store credentials in environment variables or a local environment file that is excluded by `.gitignore`. Never commit database passwords, API keys, or other secrets to GitHub.

> The exact environment variable names depend on your implementation in `app.py`. Check that file before configuring them.

### 5. Set up Ollama

Install Ollama and download the model used by the application:

```bash
ollama pull llama3.2
```

Make sure Ollama is running and the model is available before using features that require local LLM inference.

### 6. Start the application

Check the application's entry point and run the appropriate command. If `app.py` exposes a FastAPI application object named `app`, the typical command is:

```bash
uvicorn app:app --reload
```

Then open the local URL printed by the server. FastAPI applications commonly use `http://127.0.0.1:8000`, but your actual URL depends on the configuration.

**Note:** Confirm the expected startup command and configuration against the current source code before running it.

---

## 🔒 Security Notes

- Keep database credentials and secrets out of version control.
- Use strong password hashing and secure session settings.
- Validate uploaded resume files and limit accepted file types and sizes.
- Avoid logging passwords, session tokens, or sensitive resume content.
- Configure production settings separately from local development.

---

## 🛣️ Future Improvements

Potential enhancements for the project include:

- [ ] More job-source integrations.
- [ ] Advanced candidate-job ranking and filtering.
- [ ] Saved jobs and application tracking.
- [ ] Improved dashboard analytics.
- [ ] Automated testing and CI checks.
- [ ] Containerized deployment.
- [ ] More detailed explanations of candidate-job matches.

These are possible future improvements, not claims that the features are already implemented.

---

## 👨‍💻 Author

\<div align="center">

### Paramesh P

Computer Science student interested in AI, full-stack development, and building practical software.

[!\[GitHub\](https://img.shields.io/badge/GitHub-param esh25427-181717?style=for-the-badge&logo=github)](https://github.com/paramesh25427)

\</div>

---

\<div align="center">

**Built to explore how AI agents can make job discovery more useful for freshers.**

⭐ If you find this project interesting, consider starring the repository.

\</div>
