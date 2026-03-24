# 🎬 YouTube Summarizer — Generative AI Content Pipeline

A Generative AI-powered Streamlit application that converts any YouTube video into a publication-ready article webpage. The pipeline extracts the video transcript, summarizes it using Google Gemini, and generates a fully styled HTML/CSS/JS article — downloadable as a ZIP in one click.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Local Setup](#local-setup)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Limitations](#limitations)

---

## Overview

This project addresses the problem of information overload from long-form video content. Instead of watching a 2-hour podcast or tutorial, a user pastes the YouTube URL and receives:

1. A clean, structured **text summary** of the video's key insights
2. A **complete article webpage** (HTML + CSS + JS) styled like a Medium/Dev.to post, ready to publish or embed
3. A **downloadable ZIP** containing all three frontend files

The primary use case is content repurposing — particularly useful for marketing teams, technical writers, educators, and content creators who work with video-heavy sources.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI & App framework | [Streamlit](https://streamlit.io) |
| LLM | Google Gemini (`gemini-3.1-flash-lite-preview`) |
| LLM orchestration | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-google-genai`) |
| Transcript extraction | [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api) v0.6.x+ |
| Text chunking | `langchain-text-splitters` — `RecursiveCharacterTextSplitter` |
| Secrets management | Streamlit Secrets (`st.secrets`) |
| Packaging | Python `zipfile` + `tempfile` |

---

## Project Structure

```
youtube_summarizer/
├── app.py                    # Streamlit UI — layout, progress tracking, results tabs
├── pipeline.py               # Core AI pipeline — transcript, summarization, article gen, ZIP
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes secrets.toml and virtual environments
└── .streamlit/
    └── secrets.toml          # API key storage (never committed to Git)
```

**Responsibilities at a glance:**

`app.py` handles everything visual — page config, custom CSS theming, the URL input form, the 4-step progress tracker, stat cards, download buttons, and the 6-tab results panel. It imports from `pipeline.py` but contains no AI logic itself.

`pipeline.py` is the pure backend — completely decoupled from Streamlit. It handles transcript extraction, LLM instantiation, both summarization strategies, article generation with delimiter parsing, and ZIP packaging. All functions are independently testable.

---

## Pipeline Architecture

```
YouTube URL
     │
     ▼
┌─────────────────────────────┐
│  1. Transcript Extraction   │  youtube-transcript-api v0.6.x+
│     extract_video_id()      │  Parses video ID from any URL format
│     extract_transcript()    │  Fetches English captions; falls back
│                             │  to any available language
└────────────┬────────────────┘
             │  raw transcript text
             ▼
┌─────────────────────────────┐
│  2. Smart Summarisation     │  LangChain + Gemini
│     smart_summarize()       │
│                             │
│  Token estimate < 3,000     │
│  ──→ base_summarize()       │  Single LLM call with full transcript
│                             │
│  Token estimate ≥ 3,000     │
│  ──→ recursive_summarize()  │  Splits into 4,000-char chunks (200
│                             │  overlap), processes each sequentially,
│                             │  maintains a rolling summary across
└────────────┬────────────────┘  all chunks
             │  structured summary (Markdown)
             ▼
┌─────────────────────────────┐
│  3. Article Generation      │  LangChain + Gemini
│     generate_article()      │  Summary → full HTML/CSS/JS webpage
│                             │  LLM output parsed via delimiters:
│                             │  --html-- / --css-- / --js--
└────────────┬────────────────┘
             │  {"html": ..., "css": ..., "js": ...}
             ▼
┌─────────────────────────────┐
│  4. ZIP Packaging           │  Python zipfile + tempfile
│     build_zip()             │  Writes index.html, style.css,
│                             │  script.js → article_website.zip
└─────────────────────────────┘
             │  raw ZIP bytes
             ▼
        Streamlit download buttons
```

### Token routing threshold

The 3,000-token boundary (~12,000 characters) is a pragmatic cutoff. Below it, the full transcript fits comfortably in a single Gemini prompt, keeping latency minimal. Above it, sending the full transcript in one call risks hitting context limits and produces less coherent summaries, so chunked processing is used instead.

### Chunk parameters

```python
RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

`chunk_overlap=200` ensures that sentences crossing chunk boundaries are not silently dropped. The separator priority order preserves semantic units — paragraph breaks are preferred over line breaks, which are preferred over mid-sentence splits.

---

## Key Design Decisions

**Why LangChain over the Gemini SDK directly?**
LangChain's `ChatPromptTemplate` and chain (`|`) syntax keeps prompt logic declarative and easy to modify. Swapping the LLM model or changing a prompt requires touching one line, not restructuring call code.

**Why `convert_system_message_to_human=True`?**
Gemini's API does not natively support a `system` role in the same way OpenAI does. This LangChain parameter automatically converts system messages into the format Gemini expects, avoiding API errors without needing to rewrite prompts.

**Why delimiter-based parsing for the article output?**
Instructing the LLM to wrap each file's content in `--html--`, `--css--`, `--js--` delimiters is more reliable than asking for JSON (which Gemini sometimes wraps in markdown fences) and more explicit than heuristic parsing. A regex fallback is in place if the LLM ignores the delimiters entirely.

**Why `st.secrets` instead of environment variables?**
Streamlit Community Cloud has no shell environment at deploy time — `os.getenv()` returns `None`. `st.secrets` is the idiomatic, secure way to inject credentials on that platform. Locally, the same mechanism works via `.streamlit/secrets.toml`.

**Why is `pipeline.py` fully decoupled from Streamlit?**
Every function in `pipeline.py` accepts plain Python types and returns plain Python types — no `st.*` calls anywhere. This makes the pipeline independently testable, reusable in non-Streamlit contexts (e.g., a FastAPI endpoint), and much easier to reason about.

---

## Local Setup

**Prerequisites:** Python 3.9 or higher.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd youtube_summarizer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key (see Configuration below)

# 5. Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Configuration

The application reads its Gemini API key exclusively from Streamlit secrets. There is no fallback to environment variables or `.env` files.

**For local development**, edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIza..."
```

Get a free API key at [Google AI Studio](https://aistudio.google.com).

> ⚠️ `.streamlit/secrets.toml` is listed in `.gitignore` and must never be committed to version control.

**To change the model**, update the `MODEL` constant at the top of `app.py`:

```python
MODEL = "gemini-3.1-flash-lite-preview"
```

---

## Output Files

After a successful run, the following files are available for download from the results panel:

| File | Description |
|---|---|
| `article_website.zip` | Complete bundle — all three files below in a single archive |
| `index.html` | The article page, linking to `style.css` and `script.js` |
| `style.css` | All styles — layout, typography, dark/light theme toggle |
| `script.js` | Interactivity — scroll progress bar, scroll-to-top button, theme toggle |
| `summary.md` | The AI-generated summary in Markdown format |
| `transcript.txt` | The raw extracted transcript text |

---

## Limitations

- The video must have **captions enabled** — either manually uploaded or auto-generated by YouTube. Videos with no captions cannot be processed.
- **Age-restricted and private videos** are inaccessible to the transcript API regardless of account status.
- **Processing time scales with video length.** Each chunk in recursive mode requires one Gemini API call processed sequentially. A 2-hour video (~36,000 tokens, ~39 chunks) typically takes 6–10 minutes to complete.
- The quality of the generated article depends on the quality of the transcript. Auto-generated captions for heavy accents or fast speech may contain transcription errors that carry through to the summary.
- Gemini API usage is subject to your account's rate limits and quota. Free-tier accounts may hit per-minute request limits on very long recursive jobs.
