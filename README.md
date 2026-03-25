# 🎬 YouTube Summarizer — Generative AI Content Pipeline

> **YouTube URL → Transcript → AI Summary → Article Webpage → Download ZIP**

A full end-to-end Generative AI application built with **Streamlit**, **LangChain**, and **Google Gemini**. Paste any YouTube URL and the pipeline automatically extracts the transcript, distils the key insights into a structured summary, and generates a complete, styled article webpage — downloadable as a ZIP in one click.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Prompt Engineering](#prompt-engineering)
- [UI & Application Design](#ui--application-design)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations & Common Errors](#known-limitations--common-errors)
- [Local Setup](#local-setup)
- [Configuration](#configuration)
- [Output Files](#output-files)

---

## Overview

This project addresses the problem of information overload from long-form video content. Instead of watching a 2-hour podcast or tutorial, a user pastes the YouTube URL and receives:

1. A clean, structured **Markdown summary** of the video's key insights, filtered of all fluff and channel promotion
2. A **complete article webpage** (HTML + CSS + JS) styled like a Medium/Dev.to post — ready to open in a browser, publish, or embed
3. A **downloadable ZIP** containing all three frontend files plus the summary and raw transcript

The pipeline intelligently routes between two summarisation strategies based on transcript length, so both 5-minute short clips and 3-hour podcasts are handled correctly without any manual intervention from the user.

The primary real-world use case is **content repurposing** — particularly useful for marketing teams, technical writers, educators, researchers, and content creators who work with video-heavy sources and need to extract and redistribute key information quickly.

---

## Features

| Feature | Detail |
|---|---|
| **Universal URL parsing** | Supports all YouTube URL formats — `watch?v=`, `youtu.be/`, `embed/` |
| **Smart transcript extraction** | Tries English captions first, falls back to any available language |
| **Smart summarisation routing** | Automatically selects Base or Recursive mode based on transcript length |
| **Recursive chunk summarisation** | Maintains a coherent rolling summary across unlimited-length transcripts |
| **Article generation** | Produces a full HTML + CSS + JS article with dark/light theme toggle |
| **Live article preview** | Renders the generated article inside the app via an iframe |
| **5 stat cards** | Transcript tokens, summary words, reading time, pipeline mode, HTML size |
| **6 result tabs** | Article Preview, Summary, Raw Transcript, HTML, CSS, JS |
| **Individual file downloads** | Download each output file separately or as a single ZIP |
| **Session state persistence** | Results stay visible after the pipeline completes without rerunning |
| **Zero API key input** | Key is loaded from Streamlit secrets — users just paste a URL and click Run |
| **Video thumbnail preview** | Shows the video thumbnail during processing for visual confirmation |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI & App framework | [Streamlit](https://streamlit.io) `>=1.35.0` | Web interface, layout, state management |
| LLM | Google Gemini `gemini-3.1-flash-lite-preview` | Summarisation and article generation |
| LLM orchestration | [LangChain](https://www.langchain.com/) `>=0.3.0` | Prompt templating, chain composition |
| Gemini integration | `langchain-google-genai>=2.0.0` | LangChain adapter for Gemini API |
| Transcript extraction | `youtube-transcript-api>=0.6.2` | Fetching YouTube captions |
| Text chunking | `langchain-text-splitters>=0.3.0` | Splitting long transcripts into chunks |
| Secrets management | `st.secrets` | Secure API key injection |
| ZIP packaging | Python `zipfile` + `tempfile` | Bundling output files |
| Fonts | Google Fonts CDN | DM Sans, DM Serif Display, Fira Code |

---

## Project Structure

```
youtube_summarizer/
│
├── app.py                     # Streamlit UI — all visual logic, layout, state
├── pipeline.py                # Core AI pipeline — all backend logic, no Streamlit imports
├── requirements.txt           # Python dependencies with version constraints
├── .gitignore                 # Excludes secrets.toml, venv, __pycache__
│
└── .streamlit/
    └── secrets.toml           # Gemini API key (never committed to Git)
```

### Separation of concerns

The codebase is split into exactly two Python files with a strict boundary between them:

**`app.py`** owns everything visual and interactive:
- Page config, custom CSS theming, Google Fonts injection
- Sidebar layout (model badge, how-it-works steps)
- Hero banner, URL input field, Run button
- 4-step pipeline progress tracker with live status icons
- Video thumbnail preview panel
- 5 stat cards rendered after completion
- 4 individual download buttons + ZIP download
- 6-tab results panel (Preview, Summary, Transcript, HTML, CSS, JS)
- Session state persistence of all results
- Empty state UI when no video has been processed

**`pipeline.py`** owns everything computational:
- Video ID extraction from any URL format
- Transcript fetching with language fallback
- Token estimation for routing decisions
- LLM instantiation via LangChain
- Both summarisation strategies (base and recursive)
- Article generation with delimiter-based output parsing
- ZIP archive construction

`pipeline.py` contains **zero Streamlit imports**. Every function takes plain Python types as input and returns plain Python types as output, making the pipeline independently testable and reusable outside of Streamlit.

---

## Pipeline Architecture

```
YouTube URL
      │
      ▼
┌──────────────────────────────────┐
│  STEP 1 — Transcript Extraction  │
│                                  │
│  extract_video_id(url)           │  3 regex patterns cover all URL formats
│       ↓                          │
│  YouTubeTranscriptApi()          │  v0.6.x+ requires instantiation
│  .get_transcript(video_id,       │  Primary: English captions
│      languages=["en"])           │
│       ↓ (on failure)             │
│  .list(video_id)                 │  Fallback: first available language
│  → next(iter(transcript_list))   │
│  → transcript.fetch()            │
│                                  │
│  Joins all entries into a        │
│  single continuous string        │
└──────────────┬───────────────────┘
               │  raw transcript text
               ▼
┌──────────────────────────────────┐
│  STEP 2 — Smart Summarisation    │
│                                  │
│  estimate_tokens(text)           │  len(text) // 4  (approx)
│                                  │
│  tokens < 3,000  ─────────────→  base_summarize()
│  (~12,000 chars)                 │  Single LLM call, full transcript
│                                  │  in one prompt
│  tokens ≥ 3,000  ─────────────→  recursive_summarize()
│                                  │  RecursiveCharacterTextSplitter:
│                                  │    chunk_size=4000
│                                  │    chunk_overlap=200
│                                  │    separators: ["\n\n","\n",". "," ",""]
│                                  │
│                                  │  For each chunk:
│                                  │    LLM merges chunk into
│                                  │    running_summary (rolling update)
│                                  │  Progress callback updates the UI
│                                  │  per chunk (e.g. "Chunk 3 / 39")
└──────────────┬───────────────────┘
               │  structured Markdown summary
               ▼
┌──────────────────────────────────┐
│  STEP 3 — Article Generation     │
│                                  │
│  generate_article(summary, llm)  │
│                                  │
│  LLM prompt instructs Gemini to  │
│  produce output wrapped in       │
│  exact delimiters:               │
│    --html-- ... --html--         │
│    --css--  ... --css--          │
│    --js--   ... --js--           │
│                                  │
│  Regex parser extracts each      │
│  section. Falls back to treating │
│  full output as HTML if parsing  │
│  fails.                          │
│                                  │
│  Returns: {"html","css","js",    │
│            "raw"}                │
└──────────────┬───────────────────┘
               │  article dict
               ▼
┌──────────────────────────────────┐
│  STEP 4 — ZIP Packaging          │
│                                  │
│  build_zip(article)              │
│                                  │
│  tempfile.TemporaryDirectory()   │
│  Writes index.html, style.css,   │
│  script.js to temp dir           │
│                                  │
│  Auto-injects <link> and         │
│  <script> tags if the HTML       │
│  doesn't already reference       │
│  the external files              │
│                                  │
│  ZipFile(ZIP_DEFLATED)           │
│  Returns raw bytes               │
└──────────────┬───────────────────┘
               │  zip bytes + article dict + summary + transcript
               ▼
        Streamlit results panel
        (stat cards, download buttons, 6 tabs)
```

### Token routing threshold

The **3,000-token cutoff** (~12,000 characters) is the boundary between the two summarisation modes. Below it, the full transcript fits comfortably within a single Gemini context window, so one LLM call is sufficient. Above it, a single call risks hitting output quality issues and context limits, so the transcript is chunked and processed iteratively.

### Chunk overlap

The `chunk_overlap=200` parameter ensures that sentences crossing a chunk boundary are captured in both the current and the next chunk, preventing information loss at split points. The separator hierarchy `["\n\n", "\n", ". ", " ", ""]` ensures the splitter always prefers clean semantic breaks — paragraph boundaries first, then line breaks, then sentence ends — before resorting to mid-word splits.

### Processing time by video length

| Video Length | Approx. Tokens | Approx. Chunks | Approx. Time |
|---|---|---|---|
| < 10 minutes | < 3,000 | — (base mode) | 10–20 seconds |
| 15–30 minutes | 5,000–12,000 | 5–12 | 1–2 minutes |
| 45–60 minutes | 12,000–20,000 | 12–20 | 2–4 minutes |
| 90–120 minutes | 20,000–35,000 | 20–35 | 4–7 minutes |
| 2–3 hours | 35,000–55,000 | 35–55 | 7–12 minutes |

---

## Prompt Engineering

The pipeline uses five distinct prompt strings across three LangChain `ChatPromptTemplate` objects.

### Summariser prompts (base mode)

**System:** Frames the LLM as a professional technical writer focused on Medium, LinkedIn, and tech blogs. Instructs it to capture essence with zero fluff, preserve technical accuracy, and produce structured output.

**Human:** Contains two explicit sections — a `STRICT FILTER` that instructs the LLM to completely ignore introductory greetings, channel promotion (subscribe/like/comment), sponsor mentions, affiliate links, self-promotion, and filler words; and a `WHAT TO EXTRACT` section that instructs it to focus exclusively on core concepts, step-by-step processes, technical terms, tools, frameworks, code snippets, conclusions, and data. Output format is specified as: 3–5 sentence executive summary → themed bullet-point key takeaways → code in backticks → max 600 words.

### Recursive chunk summariser prompt

**Human:** Receives both the `{running_summary}` (everything processed so far) and the `{chunk}` (new content). Instructs the LLM to integrate new information, compress redundant content, and maintain a structured rolling summary under 800 words. Uses the same filter rules as the base prompt. This is the core of the recursive strategy — the summary improves and compresses with each successive chunk rather than growing linearly.

### Article generator prompts

**System:** Frames the LLM as a Senior Frontend Web Developer with 10+ years of experience producing publication-ready webpages in the style of Medium, Dev.to, Hashnode, and Substack.

**Human:** Specifies detailed design requirements — dark theme default with light/dark toggle, 750px centered editorial layout, Google Fonts (Merriweather + Poppins), hero section, heading hierarchy, styled code blocks with Fira Code, custom bullet list styling, reading time estimate, scroll-to-top button, mobile responsiveness, SEO meta tags, and a scroll-progress bar. Uses the `--html-- / --css-- / --js--` delimiter format to enable reliable regex parsing of the three output sections.

---

## UI & Application Design

### Theme & Typography

The app uses a custom **light theme** built entirely with injected CSS via `st.markdown()`, overriding Streamlit's default styling. Three Google Fonts are loaded from CDN:

- **DM Serif Display** — headings, hero title, stat values
- **DM Sans** — body text, labels, buttons, all UI copy
- **Fira Code** — code blocks in the transcript and source code tabs

The colour palette is warm off-white (`#f7f6f3`) for the main background, pure white (`#ffffff`) for cards and the sidebar, and violet (`#7c3aed`) as the primary accent — used for the Run button gradient, focus rings, active tab indicators, and token count highlights.

### Layout

The app uses Streamlit's wide layout (`layout="wide"`) with an expanded sidebar. The main content area consists of:

1. **Hero banner** — gradient background (white → lavender → warm white) with two radial glow decorations, a badge pill, serif title with gradient text, and subtitle
2. **URL input + Run button** — a 5:1 column split so the input takes most of the width
3. **Progress panel** (shown during processing) — 4-step tracker (left) and video thumbnail (right) in a 3:2 column split
4. **Results panel** (shown after completion) — stat row, download buttons, 6-tab panel

### Pipeline progress tracker

The 4 steps (Extract transcript → Summarise → Generate article → Package ZIP) each render as a `<div>` with a circular icon and label. The icon background and border change based on status:

| Status | Icon | Colours |
|---|---|---|
| `pending` | neutral | Grey background `#f3f4f6` |
| `active` | ⏳ | Violet tint `rgba(124,58,237,0.08)` + violet border |
| `done` | ✅ | Green tint `rgba(5,150,105,0.08)` + green border |
| `error` | ❌ | Red tint `rgba(220,38,38,0.08)` + red border |

Each step placeholder is updated live using `st.empty()`, so the UI reflects real-time pipeline state without a page re-render.

### Results tabs

After a successful run, 6 tabs are rendered:

| Tab | Content |
|---|---|
| 🌐 Article Preview | `st.components.v1.html()` with CSS and JS inlined into the HTML for self-contained iframe rendering |
| 📝 Summary | `st.markdown()` rendering with a `.md` download button |
| 📄 Raw Transcript | Character/word/token count header, scrollable `st.text_area`, `.txt` download |
| 🔷 HTML | Character count, scrollable `st.text_area` with the raw HTML source |
| 🎨 CSS | Character count, scrollable `st.text_area` with the raw CSS source |
| ⚙️ JS | Character count, scrollable `st.text_area` with the raw JavaScript source |

The Article Preview tab inlines the CSS and JS directly into the HTML before passing it to `st.components.v1.html()` because the iframe cannot load external `style.css` or `script.js` files from the filesystem — they don't exist as served URLs in Streamlit's environment.

---

## Key Design Decisions

**Why LangChain over calling the Gemini SDK directly?**
LangChain's `ChatPromptTemplate` and `|` chain composition keep all prompt logic declarative and modular. Changing a prompt, swapping the model, or adding an output parser requires changing one line rather than restructuring imperative call code. The `StrOutputParser()` at the end of every chain also standardises output handling across all three prompt types.

**Why `convert_system_message_to_human=True` in the LLM config?**
Gemini's API does not support a dedicated `system` role the way OpenAI's API does. Without this parameter, LangChain would send the system message in a format Gemini rejects, causing an API error. This flag instructs LangChain to automatically convert system messages into a human-turn format that Gemini accepts — no prompt rewrites needed.

**Why delimiter-based parsing (`--html-- / --css-- / --js--`) for the article output?**
Asking Gemini to return JSON is unreliable because it frequently wraps the response in markdown code fences (` ```json `) or adds explanatory text outside the JSON structure. Plain delimiters are simpler for the model to follow consistently. A regex fallback (`if not result["html"]: result["html"] = raw_output`) ensures that even if the model ignores the delimiters entirely, the app still returns something usable rather than crashing.

**Why `st.secrets` instead of `os.getenv()` or a `.env` file?**
Streamlit Community Cloud does not expose shell environment variables — `os.getenv("GEMINI_API_KEY")` returns `None` at runtime on that platform. `st.secrets` is the correct, idiomatic mechanism for credential injection on Streamlit Cloud. Locally, the exact same code path works via `.streamlit/secrets.toml`. This means one credential-loading pattern works identically in both environments with no conditional logic.

**Why is the article preview inlined rather than served as separate files?**
`st.components.v1.html()` renders content inside a sandboxed iframe. That iframe cannot make relative file requests to `style.css` or `script.js` because those files have no URL — they only exist in memory as strings. Inlining the CSS inside a `<style>` tag and the JS inside a `<script>` tag produces a fully self-contained HTML document that renders correctly in the iframe without any filesystem access.

**Why does `pipeline.py` contain zero Streamlit imports?**
Keeping the pipeline completely decoupled from Streamlit means every function (`extract_transcript`, `smart_summarize`, `generate_article`, `build_zip`) can be tested in isolation with a plain Python test runner, called from a CLI script, or exposed via a FastAPI endpoint — without any dependency on the Streamlit runtime.

---

## Known Limitations & Common Errors

### YouTube IP rate-limiting (most common)

```
Pipeline error: Failed to fetch transcript: Could not retrieve a transcript...
YouTube is blocking requests from your IP.
```

**Cause:** `youtube-transcript-api` makes HTTP requests to YouTube's servers. When deployed on Streamlit Community Cloud (which runs on AWS), YouTube detects the cloud provider IP range and applies a temporary rate-limit after a certain volume of requests. This is not a bug in the code — it is a deliberate restriction on YouTube's side.

**What is affected:** Only the shared AWS IP address assigned to Streamlit's infrastructure. No user accounts, API keys, or personal IP addresses are involved in any way.

**Fix:** Wait 4–12 hours. The rate-limit expires automatically and the app resumes working. Rebooting the app does not help — it restarts the Python process but keeps the same IP address.

### Gemini API overload (503)

```
Pipeline error: 503 UNAVAILABLE. This model is currently experiencing high demand.
```

**Cause:** Google's Gemini servers are temporarily overloaded, or the free-tier requests-per-minute quota on the API key has been exceeded due to concurrent users. 

**Fix:** Wait a few minutes and try again. Check quota usage at [aistudio.google.com](https://aistudio.google.com).

### Videos with no captions

```
Pipeline error: Transcripts are disabled for this video.
Pipeline error: No transcript found for this video.
```

**Cause:** The video owner has disabled captions, or YouTube has not generated auto-captions for the video yet (common for very new uploads).

**Fix:** Choose a video that has captions enabled. Most videos older than a few hours with English speech have auto-generated captions available.

### Age-restricted or private videos

**Cause:** `youtube-transcript-api` makes unauthenticated requests and cannot access videos that require a logged-in session.

**Fix:** The app cannot process these videos without cookie-based authentication, which is not implemented (and not recommended, as per the library's own documentation).

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

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure your API key (see Configuration section below)

# 5. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## Configuration

### API key setup

The app reads the Gemini API key **exclusively** from Streamlit secrets. There is no `.env` file fallback and no manual text input in the UI.

Edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIza..."
```

Get a free key at [Google AI Studio](https://aistudio.google.com). The free tier is sufficient for personal and course project use.

> ⚠️ **Security:** `.streamlit/secrets.toml` is listed in `.gitignore` and must never be committed to version control. If a key is accidentally pushed to a public repository, rotate it immediately at Google AI Studio.

### Changing the model

The model is set as a module-level constant at the top of `app.py`:

```python
MODEL = "gemini-3.1-flash-lite-preview"
```

Change this string to switch models. All model names must match the exact identifier used by the Gemini API.

---

## Output Files

Every successful run makes the following files available for download from the results panel:

| File | Tab | Description |
|---|---|---|
| `article_website.zip` | Downloads row | Complete bundle — index.html + style.css + script.js in one archive |
| `index.html` | Downloads row / 🔷 HTML tab | The article page, links to style.css and script.js |
| `style.css` | Downloads row / 🎨 CSS tab | All styles — editorial layout, typography, dark/light theme |
| `script.js` | Downloads row / ⚙️ JS tab | Scroll progress bar, scroll-to-top button, theme toggle |
| `summary.md` | 📝 Summary tab | AI-generated Markdown summary |
| `transcript.txt` | 📄 Raw Transcript tab | Raw extracted transcript text |

The ZIP file is the primary deliverable — unzip it anywhere and open `index.html` in a browser to view the article exactly as previewed in the app.
