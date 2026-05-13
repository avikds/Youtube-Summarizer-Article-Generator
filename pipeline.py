import re
import zipfile
import tempfile
import os

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ──────────────────────────────────────────────
# 1.  TRANSCRIPT EXTRACTION
# ──────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Parse video ID from any YouTube URL format."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def extract_transcript(url: str) -> str:
    """
    Extract the full transcript from a YouTube video.
    Compatible with youtube-transcript-api v0.6.x+ where get_transcript()
    and list() are instance methods — requires instantiation first.
    """
    video_id = extract_video_id(url)
    try:
        # v0.6.x+: must instantiate the class before calling any method
        ytt = YouTubeTranscriptApi()

        # Primary: fetch English transcript directly
        try:
            entries = ytt.get_transcript(video_id, languages=["en"])
        except Exception:
            # Fallback: list all available transcripts and pick the first one
            transcript_list = ytt.list(video_id)
            transcript = next(iter(transcript_list))
            entries = list(transcript.fetch())

        # entries are dicts with a "text" key
        full_text = " ".join(
            (e["text"] if isinstance(e, dict) else e.text) for e in entries
        )
        return full_text.strip()

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video.")
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {str(e)}")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# ──────────────────────────────────────────────
# 2.  PROMPTS
# ──────────────────────────────────────────────

SUMMARIZER_SYSTEM = """You are a professional technical writer and content strategist specializing in \
distilling complex video content into clear, actionable written content for Medium, LinkedIn, and tech blogs.

Your summaries:
- Capture the ESSENCE of what was taught or discussed
- Contain zero fluff — every sentence earns its place
- Preserve technical accuracy and code/commands exactly
- Follow a structured flow that guides readers naturally
"""

SUMMARIZER_HUMAN = """Transform the following YouTube transcript into a **concise, high-quality summary**.

**STRICT FILTER — IGNORE completely**:
- All introductory greetings (welcome, hi, hello, "in this video we will...")
- Channel promotion (subscribe, like, comment, follow, share)
- Sponsor mentions, affiliate links, discount codes
- Self-promotion (my course, my Discord, my newsletter, check description)
- Off-topic banter, filler words, repetition

**WHAT TO EXTRACT**:
- Core concepts and key ideas explained
- Step-by-step processes or tutorials
- Technical terms, tools, frameworks, commands, and code snippets
- Important conclusions, insights, or recommendations
- Any data, statistics, or benchmarks mentioned

**OUTPUT FORMAT**:
- Write a 3–5 sentence executive summary at the top
- Then bullet-point the key takeaways grouped by theme
- Preserve any code or commands in backticks
- Keep total length under 600 words

TRANSCRIPT:
{transcript}
"""

ARTICLE_SYSTEM = """You are a Senior Frontend Web Developer (10+ years) AND an expert technical writer. \
You produce publication-ready, visually stunning article webpages — think Medium, Dev.to, Hashnode, Substack. \
Your HTML is semantic, your CSS is modern, and your JavaScript adds tasteful interactivity.
"""

ARTICLE_HUMAN = """Create a **complete, production-ready article webpage** based on the summary below.

**DESIGN REQUIREMENTS**:
- Dark theme by default with a light/dark toggle button (moon/sun icon)
- Clean editorial layout (max-width 750px centered, like Medium)
- Google Fonts: Merriweather for body text, Poppins for headings
- Hero section with the article title and a subtitle
- Proper heading hierarchy (h1 → h2 → h3)
- Styled code blocks with dark background and monospace font (Fira Code or similar)
- Bullet lists with custom left-border accent styling
- Estimated reading time displayed near the title
- Smooth scroll-to-top floating button (appears after scrolling 300px)
- Fully responsive and mobile-friendly
- SEO meta tags (title, description, og:title, og:description)
- Progress bar at the top that fills as the user scrolls

**CONTENT TO USE**:
{article_content}

**MANDATORY OUTPUT FORMAT** — use these EXACT delimiters, nothing outside them:

--html--
[complete HTML here, linking to style.css and script.js]
--html--

--css--
[complete CSS here]
--css--

--js--
[complete JavaScript here]
--js--
"""

CHUNK_SUMMARIZER_HUMAN = """You are building a rolling summary of a long video transcript.

**Current running summary** (everything processed so far):
{running_summary}

**New transcript chunk** to incorporate:
{chunk}

**YOUR TASK**:
- Integrate key new information from the chunk into the running summary
- Remove or compress redundant information
- Keep the summary focused, structured, and under 800 words
- IGNORE all channel promotions, subscriptions, sponsors, greetings
- PRESERVE all technical details, code, steps, and insights

Output ONLY the updated summary — no preamble, no commentary.
"""


# ──────────────────────────────────────────────
# 3.  LLM FACTORY
# ──────────────────────────────────────────────

def get_llm(api_key: str, model: str = "gemini-3.1-flash-lite") -> ChatGoogleGenerativeAI:
    """Instantiate a Gemini LLM via LangChain."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=1.0,
    )


# ──────────────────────────────────────────────
# 4.  SUMMARIZATION PIPELINES
# ──────────────────────────────────────────────

def base_summarize(transcript: str, llm) -> str:
    """Single-pass summarization for short transcripts (< 3000 tokens)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZER_SYSTEM),
        ("human", SUMMARIZER_HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": transcript})


def recursive_summarize(transcript: str, llm, progress_callback=None) -> str:
    """
    Chunk-based recursive summarization for long transcripts.
    Maintains a rolling summary across all chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(transcript)
    running_summary = ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZER_SYSTEM),
        ("human", CHUNK_SUMMARIZER_HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total)
        running_summary = chain.invoke({
            "running_summary": running_summary,
            "chunk": chunk,
        })

    return running_summary


def smart_summarize(transcript: str, llm, progress_callback=None) -> str:
    """
    Routes to base_summarize or recursive_summarize based on token count.
    Threshold: 3000 tokens (~12,000 characters).
    """
    token_count = estimate_tokens(transcript)
    if token_count < 3000:
        return base_summarize(transcript, llm)
    else:
        return recursive_summarize(transcript, llm, progress_callback)


# ──────────────────────────────────────────────
# 5.  ARTICLE GENERATION
# ──────────────────────────────────────────────

def generate_article(summary: str, llm) -> dict:
    """
    Convert a summary into a full HTML/CSS/JS article webpage.
    Returns a dict with keys: html, css, js, raw
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", ARTICLE_SYSTEM),
        ("human", ARTICLE_HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"article_content": summary})

    result = {"raw": raw_output, "html": "", "css": "", "js": ""}

    def extract_section(text: str, tag: str) -> str:
        pattern = rf"--{tag}--\s*([\s\S]*?)\s*--{tag}--"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    result["html"] = extract_section(raw_output, "html")
    result["css"] = extract_section(raw_output, "css")
    result["js"] = extract_section(raw_output, "js")

    # Fallback if parsing failed
    if not result["html"]:
        result["html"] = raw_output

    return result


# ──────────────────────────────────────────────
# 6.  ZIP EXPORT
# ──────────────────────────────────────────────

def build_zip(article: dict) -> bytes:
    """
    Bundle index.html, style.css, and script.js into a ZIP archive.
    Returns the ZIP as raw bytes for Streamlit download.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "index.html")
        css_path  = os.path.join(tmpdir, "style.css")
        js_path   = os.path.join(tmpdir, "script.js")
        zip_path  = os.path.join(tmpdir, "article_website.zip")

        html_content = article["html"]
        # Inject external file refs if not already present
        if article["css"] and "style.css" not in html_content:
            html_content = html_content.replace(
                "</head>", '  <link rel="stylesheet" href="style.css">\n</head>'
            )
        if article["js"] and "script.js" not in html_content:
            html_content = html_content.replace(
                "</body>", '  <script src="script.js"></script>\n</body>'
            )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(article.get("css", ""))
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(article.get("js", ""))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(html_path, "index.html")
            zf.write(css_path,  "style.css")
            zf.write(js_path,   "script.js")

        with open(zip_path, "rb") as f:
            return f.read()


# ──────────────────────────────────────────────
# 7.  FULL PIPELINE ENTRY POINT
# ──────────────────────────────────────────────

def run_pipeline(
    youtube_url: str,
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    progress_callback=None,
) -> dict:
    """
    Full end-to-end pipeline:
      URL → Transcript → Summary → Article (HTML/CSS/JS) → ZIP bytes

    Returns a dict with:
        transcript, summary, article (html/css/js), zip_bytes, token_count, mode
    """
    llm = get_llm(api_key, model)

    # Step 1: Extract transcript
    transcript = extract_transcript(youtube_url)
    token_count = estimate_tokens(transcript)
    mode = "base" if token_count < 3000 else "recursive"

    # Step 2: Smart summarization (routes automatically)
    summary = smart_summarize(transcript, llm, progress_callback)

    # Step 3: Article generation
    article = generate_article(summary, llm)

    # Step 4: Build ZIP
    zip_bytes = build_zip(article)

    return {
        "transcript": transcript,
        "summary": summary,
        "article": article,
        "zip_bytes": zip_bytes,
        "token_count": token_count,
        "mode": mode,
    }
