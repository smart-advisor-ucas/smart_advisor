# Smart Advisor for DS and AI Students

> A voice-enabled Arabic RAG-based academic advisor for new and prospective Data Science and AI students at the University College of Applied Sciences (UCAS).

![Status](https://img.shields.io/badge/status-Completed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

---
## 📖 About

The Smart Advisor is a graduation project that helps students who just graduated from high school get instant, accurate advice about the Data Science and AI specialization at UCAS — admission requirements, study plan, career paths, and scholarships. Unlike generic AI tools (ChatGPT, Gemini), our system grounds every answer in **official UCAS documents** — eliminating hallucination — and supports both **text and Arabic voice** input.

**Phase 1 (Completed):** Research, literature review, methodology design.

**Phase 2 (Completed):** Implementation, evaluation, and deployment. The system is fully built, tested, and merged into `main`.

---

## 🧠 How It Works

```
Student speaks/types question
        ↓
   Speech-to-Text (Whisper Arabic)
        ↓
   RAG Pipeline (retrieval + generation)
        ↓
   Text-to-Speech (Arabic TTS)
        ↓
   Student receives spoken/text answer
```

Every step is grounded in official UCAS documents stored in a vector database (ChromaDB). If the system can't find a confident answer, it routes the question to a human advisor instead of guessing.

---

## 👥 Team

| Member | Role | Focus |
|---|---|---|
| **Fatma Alzahraa Alhabbash** | Project Lead + Backend Architect | RAG pipeline · retrieval logic · LLM integration · fallback mechanism |
| **Roaa Alhaddad** | Data Engineer | Data collection for the knowledge base · FAQ surveys · document processing |
| **Saja Abdalaal** | Voice & NLP Engineer | STT · TTS · NLP preprocessing |
| **Shahd Ethalathini** | Frontend + Knowledge Base Engineer | UI · KB chunking · embedding generation · storing embeddings in ChromaDB |

**Supervisor:** Dr. Sanaa Al-Sayegh
**Institution:** University College of Applied Sciences — Gaza

---

## 🗂️ Repository Structure

```
smart-advisor/
├── docs/                    Documentation, reports
├── data/                    Raw and processed data (mostly gitignored)
├── src/
│   ├── knowledge_base/      Document processing → chunking → embeddings → ChromaDB
│   ├── rag/                 Retrieval and generation pipeline
│   ├── voice/               STT and TTS modules
│   ├── ui/                  FastAPI backend + Gradio interface
│   └── utils/               Shared helpers (logging, config)
├── notebooks/               Jupyter experiments
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- ~5GB free disk space (for model weights)
- (Optional) GPU for faster STT inference

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/smart-advisor-ucas/smart_advisor.git
   cd smart-advisor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Open .env and fill in your API keys
   ```
5. **Add the ChromaDB knowledge base**
   The `data/chroma_db/` folder is **not tracked in GitHub** (it's git-ignored). Create it and populate it before running the backend.
   Without this step, the backend will start but retrieval will fail because the collection won't exist.

7. **Run the backend**
   ```bash
   uvicorn src.ui.main:app --reload --port 8000
   ```
   Swagger docs are available at `http://127.0.0.1:8000/docs`.

6. **Run the interface**
   ```bash
   cd src/ui/frontend
   npm install
   npm run dev
   ```
   The React frontend talks to the backend over the `/chat`, `/chat/reset`, and `/chat/history/{session_id}` endpoints.

---

## Branch Strategy
 
We use a simple branching model:
 
- **`main`** — protected, only updated via pull requests
- **`integration/merge-all`** — integration branch where all feature branches merge first
- **`feature/rag`** — backend RAG pipeline work
- **`feature/frontend-ui`** — UI work
- **`feature/voice`** — STT/TTS work
  
---

## ✅ Testing

- **Unit testing** was carried out at the end of each development phase, on each component in isolation (retrieval, metadata filtering, fallback logic, profile extraction, etc.) before it was merged.
- **Integration/overall testing** was performed after all feature branches (`feature/rag`, `feature/frontend-ui`, `feature/voice`) were merged into `main`, to validate the end-to-end flow from onboarding through retrieval to fallback escalation.
- Test cases and results are tracked under `tests/`.

---

## 📚 Documentation

- [Phase 1 Report](docs/phase1_report.pdf)
- [Architecture Overview](docs/architecture.md)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Dr. Sanaa Al-Sayegh, our supervisor, for invaluable guidance throughout the project.
- The Department of Computer Engineering at UCAS for academic support.
- The open-source NLP community whose tools make this project possible.

---

*Built with ❤️ by the Smart Advisor team — UCAS, Gaza, 2025-2026*
