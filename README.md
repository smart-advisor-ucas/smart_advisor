# Smart Advisor for DS and AI Students

> A voice-enabled Arabic RAG-based academic advisor for new and prospective Data Science and AI students at the University College of Applied Sciences (UCAS).

[![Status](https://img.shields.io/badge/status-Phase%202%20In%20Progress-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()

---

## 📖 About

The Smart Advisor is a graduation project that helps DS/AI students at UCAS get instant, accurate answers to their academic questions. Unlike generic AI tools (ChatGPT, Gemini), our system grounds every answer in **official UCAS documents** — eliminating hallucination — and supports both **text and Arabic voice** input.

**Phase 1 (Complete):** Research, literature review, methodology design.
**Phase 2 (In Progress):** Implementation, evaluation, and deployment.

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
| **Fatema Alhabbash** | Project Lead + Backend Architect | RAG Pipeline · LLM Integration |
| **Roaa Alhaddad** | Data Engineer | Document Processing · Embeddings · ChromaDB |
| **Saja Abdalaal** | Voice & NLP Engineer | STT · TTS · NLP Preprocessing |
| **Shahd Ethalathini** | Frontend + Evaluation Lead | UI · Metrics · Testing |

**Supervisor:** Dr. Sanaa Al-Sayegh
**Institution:** University College of Applied Sciences — Gaza

---

## 🗂️ Repository Structure

```
smart-advisor/
├── docs/                    Documentation, reports, meeting notes
├── data/                    Raw and processed data (mostly gitignored)
├── src/
│   ├── knowledge_base/      Document processing → embeddings → ChromaDB
│   ├── rag/                 Retrieval and generation pipeline
│   ├── voice/               STT and TTS modules
│   ├── ui/                  Streamlit/Gradio interface
│   └── utils/               Shared helpers (logging, config)
├── notebooks/               Jupyter experiments
├── tests/                   Automated tests
└── scripts/                 One-off setup scripts
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
   git clone https://github.com/[YOUR-ORG-NAME]/smart-advisor.git
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

5. **Run the app**
   ```bash
   streamlit run src/ui/app.py
   ```

---

## 🌳 Branch Strategy

We use a simple branching model:

- **`main`** — protected, only updated via pull requests
- **`dev`** — integration branch where features merge first
- **`feature/<name>`** — your working branch for any new work

**Workflow:**
1. Create your feature branch: `git checkout -b feature/stt-whisper-integration`
2. Commit your changes locally
3. Push to GitHub: `git push -u origin feature/stt-whisper-integration`
4. Open a Pull Request to `dev`
5. Get at least one teammate's review
6. Merge to `dev`
7. When `dev` is stable, merge to `main`

**Never push directly to `main` or `dev`.**

---

## 📝 Commit Message Convention

Use clear, prefixed commit messages:

- `feat:` new feature (e.g., `feat: add Whisper STT integration`)
- `fix:` bug fix (e.g., `fix: handle empty audio input gracefully`)
- `docs:` documentation only (e.g., `docs: update README setup steps`)
- `test:` add or update tests
- `refactor:` code restructuring without behavior change
- `chore:` housekeeping (dependencies, config, etc.)

---

## 📚 Documentation

- [Phase 1 Report](docs/phase1_report.pdf)
- [Architecture Overview](docs/architecture.md)
- [Meeting Notes](docs/meeting_notes/)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Dr. Sanaa Al-Sayegh, our supervisor, for invaluable guidance throughout Phase 1.
- The Department of Computer Engineering at UCAS for academic support.
- The open-source NLP community whose tools make this project possible.

---

*Built with ❤️ by the Smart Advisor team — UCAS, Gaza, 2025-2026*
