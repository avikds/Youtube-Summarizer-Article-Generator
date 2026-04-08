import streamlit as st
from pipeline import run_pipeline, extract_video_id

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="YT Summarizer · GenAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# LOAD API KEY FROM STREAMLIT SECRETS
# ──────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "⚠️  **API key not found.** "
        "Please add `GEMINI_API_KEY` to your Streamlit secrets. "
        "See the README for instructions."
    )
    st.stop()

MODEL = "gemini-3.1-flash-lite-preview"

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display:ital@0;1&family=Fira+Code:wght@400;500&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f7f6f3 !important;
    color: #1a1a2e !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e2da !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 24px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

header {
    background: transparent !important;
    height: 0px !important;
}

[data-testid="collapsedControl"] {
    top: 10px !important;
}

[data-testid="stDecoration"] { display: none; }

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    color: #1a1a2e !important;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #ffffff 0%, #f0edff 50%, #fff5f0 100%);
    border: 1px solid #e5e2da;
    border-radius: 20px;
    padding: 48px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 40px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(251,113,81,0.08) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.22);
    color: #7c3aed;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.7rem;
    font-weight: 400;
    color: #1a1a2e !important;
    margin: 0 0 10px 0;
    line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(135deg, #7c3aed, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: #6b7280;
    font-weight: 400;
    line-height: 1.65;
    max-width: 580px;
    margin: 0;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #d1cfc8 !important;
    border-radius: 12px !important;
    color: #1a1a2e !important;
    caret-color: #000000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #b0ada6 !important; }
.stTextInput > label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.28) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.38) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: #ffffff !important;
    border: 1.5px solid #d1cfc8 !important;
    color: #374151 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.stDownloadButton > button:hover {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    background: rgba(124,58,237,0.04) !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.1) !important;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 20px 0;
    flex-wrap: wrap;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e2da;
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1;
    min-width: 130px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-label {
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
    margin-bottom: 5px;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a1a2e;
    font-family: 'DM Serif Display', serif;
}
.stat-accent { color: #7c3aed; }
.stat-green  { color: #059669; }
.stat-red    { color: #dc2626; }

/* ── Pipeline steps ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #f0ede8;
}
.step-row:last-child { border-bottom: none; }
.step-icon {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.88rem;
    flex-shrink: 0;
}
.step-pending { background: #f3f4f6; color: #9ca3af; border: 1px solid #e5e7eb; }
.step-active  { background: rgba(124,58,237,0.08); color: #7c3aed; border: 1px solid rgba(124,58,237,0.3); }
.step-done    { background: rgba(5,150,105,0.08);  color: #059669; border: 1px solid rgba(5,150,105,0.3); }
.step-error   { background: rgba(220,38,38,0.08);  color: #dc2626; border: 1px solid rgba(220,38,38,0.3); }
.step-text    { font-size: 0.88rem; color: #4b5563; }
.step-text strong { color: #1a1a2e; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 2px !important;
    border-bottom: 2px solid #e5e2da !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #7c3aed !important;
    border-bottom: 2px solid #7c3aed !important;
    font-weight: 700 !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #d1cfc8 !important;
    color: #1a1a2e !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 10px !important;
}

/* ── Sidebar ── */
.sidebar-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
    font-weight: 700;
    margin-bottom: 10px;
}
.sidebar-info-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 7px 0;
    border-bottom: 1px solid #f0ede8;
    font-size: 0.83rem;
    color: #4b5563;
}
.sidebar-info-row:last-child { border-bottom: none; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
    border-radius: 4px !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d1cfc8 !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f7f6f3; }
::-webkit-scrollbar-thumb { background: #d1cfc8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #b0ada6; }

/* ── Markdown in tabs ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] code {
    color: #374151 !important;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())

def reading_time(text: str) -> str:
    minutes = max(1, word_count(text) // 200)
    return f"{minutes} min"

def get_thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

def update_step(placeholder, icon: str, label: str, status: str = "pending"):
    placeholder.markdown(f"""
    <div class="step-row">
        <div class="step-icon step-{status}">{icon}</div>
        <div class="step-text"><strong>{label}</strong></div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 28px 0;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;
                    color:#1a1a2e;line-height:1.2;">
            🎬 YT Summarizer
        </div>
        <div style="font-size:0.78rem;color:#9ca3af;margin-top:6px;font-weight:500;">
            Powered by Google Gemini
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model badge (read-only display, not a user input)
    st.markdown("""
    <div style="background:#f0edff;border:1px solid #ddd6fe;border-radius:10px;
                padding:14px 16px;margin-bottom:20px;">
        <div class="sidebar-title">🤖 Active Model</div>
        <div style="font-size:0.88rem;font-weight:700;color:#7c3aed;
                    font-family:'Fira Code',monospace;">
            gemini-3.1-flash-lite-preview
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📖 How It Works</div>', unsafe_allow_html=True)

    steps_info = [
        ("🔗", "Paste a YouTube URL"),
        ("📄", "Transcript is extracted"),
        ("✂️", "Smart routing: base or recursive"),
        ("🤖", "AI summarizes content"),
        ("🌐", "Article webpage generated"),
        ("📦", "Download as ZIP"),
    ]
    for icon, text in steps_info:
        st.markdown(f"""
        <div class="sidebar-info-row">
            <span>{icon}</span><span>{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#9ca3af;line-height:1.7;text-align:center;">
        Built with Streamlit · LangChain<br>
        YouTube Transcript API · Gemini API
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN CONTENT — Hero
# ──────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 Generative AI · Content Pipeline</div>
    <h1 class="hero-title">YouTube → <span>Article Generator</span></h1>
    <p class="hero-sub">
        Drop any YouTube URL. Gemini AI extracts the transcript, distils the key insights,
        and crafts a publication‑ready article webpage — download and publish instantly.
    </p>
</div>
""", unsafe_allow_html=True)

# ── URL Input + Run button ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    youtube_url = st.text_input(
        "YOUTUBE URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run", use_container_width=True)


# ──────────────────────────────────────────────
# PIPELINE EXECUTION
# ──────────────────────────────────────────────

if run_btn:
    if not youtube_url.strip():
        st.error("⚠️  Please enter a YouTube URL.")
        st.stop()

    try:
        video_id = extract_video_id(youtube_url)
    except ValueError:
        st.error("❌  Invalid YouTube URL. Please check the link and try again.")
        st.stop()

    # ── Progress layout: steps (left) + thumbnail (right) ──
    left_col, right_col = st.columns([3, 2])

    with right_col:
        st.markdown(f"""
        <div style="border:1px solid #e5e2da;border-radius:14px;overflow:hidden;
                    box-shadow:0 4px 16px rgba(0,0,0,0.06);margin-top:8px;">
            <img src="{get_thumbnail_url(video_id)}"
                 style="width:100%;display:block;object-fit:cover;"
                 onerror="this.style.display='none'">
            <div style="padding:12px 14px;background:#ffffff;">
                <div style="font-size:0.73rem;color:#9ca3af;word-break:break-all;">
                    {youtube_url}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with left_col:
        st.markdown("""
        <div style="font-size:0.75rem;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.09em;font-weight:700;margin-bottom:8px;">
            Pipeline Progress
        </div>
        """, unsafe_allow_html=True)

        ph1 = st.empty()
        ph2 = st.empty()
        ph3 = st.empty()
        ph4 = st.empty()
        progress_bar  = st.progress(0)
        status_text   = st.empty()

    # Initialise all steps as pending
    update_step(ph1, "📄", "Extract transcript",    "pending")
    update_step(ph2, "🤖", "Summarise content",      "pending")
    update_step(ph3, "🌐", "Generate article",       "pending")
    update_step(ph4, "📦", "Package ZIP",            "pending")

    try:
        from pipeline import (
            extract_transcript, estimate_tokens, get_llm,
            smart_summarize, generate_article, build_zip,
        )

        # ── Step 1: Transcript ──
        update_step(ph1, "⏳", "Extracting transcript…", "active")
        progress_bar.progress(8)
        status_text.markdown(
            "<span style='color:#7c3aed;font-size:0.83rem;'>Connecting to YouTube…</span>",
            unsafe_allow_html=True,
        )

        transcript  = extract_transcript(youtube_url)
        token_count = estimate_tokens(transcript)
        mode        = "base" if token_count < 3000 else "recursive"

        update_step(ph1, "✅", f"Transcript ready — {token_count:,} tokens ({mode} mode)", "done")
        progress_bar.progress(25)

        # ── Step 2: Summarise ──
        update_step(ph2, "⏳", "Summarising content…", "active")
        status_text.markdown(
            f"<span style='color:#7c3aed;font-size:0.83rem;'>Using <strong>{mode}</strong> summarisation…</span>",
            unsafe_allow_html=True,
        )

        llm           = get_llm(GEMINI_API_KEY, MODEL)
        chunk_ph      = st.empty()

        def chunk_callback(current, total):
            pct = 25 + int((current / total) * 35)
            progress_bar.progress(pct)
            chunk_ph.markdown(
                f"<span style='color:#6b7280;font-size:0.78rem;'>Chunk {current} / {total}</span>",
                unsafe_allow_html=True,
            )

        summary = smart_summarize(
            transcript, llm,
            chunk_callback if mode == "recursive" else None,
        )
        chunk_ph.empty()

        update_step(ph2, "✅", f"Summary done — {word_count(summary):,} words", "done")
        progress_bar.progress(65)

        # ── Step 3: Article ──
        update_step(ph3, "⏳", "Building article webpage…", "active")
        status_text.markdown(
            "<span style='color:#7c3aed;font-size:0.83rem;'>Generating HTML / CSS / JS…</span>",
            unsafe_allow_html=True,
        )

        article = generate_article(summary, llm)

        update_step(ph3, "✅", "Article generated", "done")
        progress_bar.progress(88)

        # ── Step 4: ZIP ──
        update_step(ph4, "⏳", "Packaging ZIP archive…", "active")
        zip_bytes = build_zip(article)

        update_step(ph4, "✅", "ZIP ready!", "done")
        progress_bar.progress(100)
        status_text.markdown(
            "<span style='color:#059669;font-size:0.9rem;font-weight:700;'>✅ All done! Scroll down to view your results.</span>",
            unsafe_allow_html=True,
        )

        # Persist to session state
        st.session_state["result"] = {
            "transcript": transcript,
            "summary":    summary,
            "article":    article,
            "zip_bytes":  zip_bytes,
            "token_count": token_count,
            "mode":       mode,
            "url":        youtube_url,
            "video_id":   video_id,
        }

    except Exception as exc:
        update_step(ph1, "❌", "Error", "error")
        st.error(f"**Pipeline error:** {exc}")
        st.info(
            "💡 Tips: Make sure the video has captions enabled. "
            "Age-restricted or private videos cannot be processed."
        )
        st.stop()


# ──────────────────────────────────────────────
# RESULTS PANEL
# ──────────────────────────────────────────────

if "result" in st.session_state:
    res = st.session_state["result"]

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:2px solid #e5e2da;margin:8px 0 24px 0;'>",
        unsafe_allow_html=True,
    )

    # ── Stats ──
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Transcript Tokens</div>
            <div class="stat-value stat-accent">{res['token_count']:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Summary Words</div>
            <div class="stat-value stat-green">{word_count(res['summary']):,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Reading Time</div>
            <div class="stat-value">{reading_time(res['summary'])}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Pipeline Mode</div>
            <div class="stat-value" style="font-size:0.95rem;text-transform:capitalize;">
                {"🔁 Recursive" if res['mode'] == 'recursive' else "⚡ Base"}
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">HTML Size</div>
            <div class="stat-value">{max(1, len(res['article']['html']) // 1024)} KB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download buttons ──
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af;text-transform:uppercase;"
        "letter-spacing:.09em;font-weight:700;margin-bottom:10px;'>Downloads</div>",
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("📦 Full ZIP",    data=res["zip_bytes"],
            file_name="article_website.zip", mime="application/zip",
            use_container_width=True)
    with d2:
        st.download_button("🌐 index.html",  data=res["article"]["html"].encode(),
            file_name="index.html",          mime="text/html",
            use_container_width=True)
    with d3:
        st.download_button("🎨 style.css",   data=res["article"]["css"].encode(),
            file_name="style.css",           mime="text/css",
            use_container_width=True)
    with d4:
        st.download_button("⚙️ script.js",   data=res["article"]["js"].encode(),
            file_name="script.js",           mime="text/javascript",
            use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    t_preview, t_summary, t_transcript, t_html, t_css, t_js = st.tabs([
        "🌐 Article Preview",
        "📝 Summary",
        "📄 Raw Transcript",
        "🔷 HTML",
        "🎨 CSS",
        "⚙️ JS",
    ])

    # Preview — inline CSS+JS for iframe
    with t_preview:
        combined = res["article"]["html"]
        if res["article"]["css"]:
            combined = combined.replace(
                '<link rel="stylesheet" href="style.css">',
                f'<style>{res["article"]["css"]}</style>',
            )
            if "<style>" not in combined:
                combined = combined.replace(
                    "</head>", f'<style>\n{res["article"]["css"]}\n</style>\n</head>'
                )
        if res["article"]["js"]:
            combined = combined.replace(
                '<script src="script.js"></script>',
                f'<script>{res["article"]["js"]}</script>',
            )
            if "<script>" not in combined:
                combined = combined.replace(
                    "</body>", f'<script>\n{res["article"]["js"]}\n</script>\n</body>'
                )
        st.markdown(
            "<div style='font-size:0.8rem;color:#6b7280;margin-bottom:10px;'>"
            "Live preview of your generated article webpage.</div>",
            unsafe_allow_html=True,
        )
        st.components.v1.html(combined, height=820, scrolling=True)

    with t_summary:
        st.markdown(
            "<div style='font-size:0.8rem;color:#6b7280;margin-bottom:12px;'>"
            "AI-generated summary — the content used to write the article.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(res["summary"])
        st.download_button(
            "⬇️ Download Summary (.md)",
            data=res["summary"].encode(),
            file_name="summary.md",
            mime="text/markdown",
        )

    with t_transcript:
        wc = word_count(res["transcript"])
        st.markdown(
            f"<div style='font-size:0.8rem;color:#6b7280;margin-bottom:10px;'>"
            f"{len(res['transcript']):,} characters · {wc:,} words · "
            f"{res['token_count']:,} tokens</div>",
            unsafe_allow_html=True,
        )
        st.text_area("Transcript", value=res["transcript"], height=420,
                     label_visibility="collapsed")
        st.download_button(
            "⬇️ Download Transcript (.txt)",
            data=res["transcript"].encode(),
            file_name="transcript.txt",
            mime="text/plain",
        )

    with t_html:
        st.markdown(
            f"<div style='font-size:0.78rem;color:#6b7280;margin-bottom:8px;'>"
            f"{len(res['article']['html']):,} characters</div>",
            unsafe_allow_html=True,
        )
        st.text_area("HTML", value=res["article"]["html"], height=520,
                     label_visibility="collapsed")

    with t_css:
        st.markdown(
            f"<div style='font-size:0.78rem;color:#6b7280;margin-bottom:8px;'>"
            f"{len(res['article']['css']):,} characters</div>",
            unsafe_allow_html=True,
        )
        st.text_area("CSS", value=res["article"]["css"], height=520,
                     label_visibility="collapsed")

    with t_js:
        st.markdown(
            f"<div style='font-size:0.78rem;color:#6b7280;margin-bottom:8px;'>"
            f"{len(res['article']['js']):,} characters</div>",
            unsafe_allow_html=True,
        )
        st.text_area("JavaScript", value=res["article"]["js"], height=520,
                     label_visibility="collapsed")

else:
    # ── Empty state ──
    st.markdown("""
    <div style="
        text-align:center;padding:72px 24px;
        background:#ffffff;
        border:2px dashed #e5e2da;
        border-radius:20px;margin-top:12px;
        box-shadow:0 2px 12px rgba(0,0,0,0.04);
    ">
        <div style="font-size:3.2rem;margin-bottom:18px;">🎬</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;
                    color:#1a1a2e;margin-bottom:10px;">
            No video processed yet
        </div>
        <div style="font-size:0.9rem;color:#6b7280;max-width:420px;
                    margin:0 auto;line-height:1.7;">
            Paste a YouTube URL above and click
            <strong style="color:#7c3aed;">▶ Run</strong>
            to start the pipeline. The API key is already configured — just run it!
        </div>
        <div style="display:flex;gap:12px;justify-content:center;
                    flex-wrap:wrap;margin-top:28px;">
            <div style="background:#f7f5ff;border:1px solid #ddd6fe;border-radius:10px;
                        padding:8px 18px;font-size:0.8rem;color:#7c3aed;font-weight:600;">
                ⚡ Short videos → single-pass
            </div>
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:8px 18px;font-size:0.8rem;color:#059669;font-weight:600;">
                🔁 Long videos → recursive
            </div>
            <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;
                        padding:8px 18px;font-size:0.8rem;color:#ea580c;font-weight:600;">
                📦 Download as ZIP
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)