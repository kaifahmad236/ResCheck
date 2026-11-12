"""
app.py - Main Streamlit UI for ResCheck: AI-Based Resume & Job Description Matching System
"""

import os
import sys

# Configure clean Streamlit runtime environment (suppress network URL and inspection overhead)
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import streamlit as st

# Direct execution handler: runs clean streamlit when invoked with `python app.py`
if __name__ == "__main__" and hasattr(st, "runtime") and not st.runtime.exists():
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        __file__,
        "--server.address=localhost",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--server.headless=true",
    ]
    sys.exit(stcli.main())

# Ensure project root is in system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.parser import ResumeParser
from backend.matcher import ResumeMatcher
from backend.scorer import ResumeScorer

# ─────────────────────────────────────────────
# Page Configuration & Metadata
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResCheck — AI Resume & Job Fit Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# Cached NLP Engine Loader
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_matcher() -> ResumeMatcher:
    return ResumeMatcher()


@st.cache_resource(show_spinner=False)
def get_parser() -> ResumeParser:
    return ResumeParser()


@st.cache_resource(show_spinner=False)
def get_scorer() -> ResumeScorer:
    return ResumeScorer()


# ─────────────────────────────────────────────
# Executive Dark Theme Stylesheet
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E2E8F0;
    }

    /* Background Canvas */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #172033 0%, #0B0F19 65%, #070A10 100%);
        min-height: 100vh;
    }

    /* Brand Header */
    .brand-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2.2rem 1rem 1.4rem 1rem;
        text-align: center;
    }
    .logo-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.8rem;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .logo-badge span {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }
    .logo-badge .badge-tag {
        background: linear-gradient(135deg, #2563EB, #4F46E5);
        color: #FFFFFF;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.7rem;
    }
    .main-title-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.85rem;
        margin: 0;
    }
    .main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        margin: 0;
    }
    .main-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.6rem;
        max-width: 620px;
        line-height: 1.5;
    }

    /* Cards & Containers */
    .card {
        background: rgba(19, 27, 46, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
    }
    .card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Score Circle Dial */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.2rem;
    }
    .score-circle {
        width: 154px;
        height: 154px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        background: radial-gradient(circle, rgba(15, 23, 42, 0.9) 60%, rgba(30, 41, 59, 0.4) 100%);
        border: 3px solid rgba(255, 255, 255, 0.1);
        position: relative;
    }
    .score-number {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .score-label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-top: 0.25rem;
    }

    /* Score Level Styling */
    .score-excellent { border-color: #10B981; }
    .score-excellent .score-number { color: #34D399; }
    .score-good { border-color: #3B82F6; }
    .score-good .score-number { color: #60A5FA; }
    .score-average { border-color: #F59E0B; }
    .score-average .score-number { color: #FBBF24; }
    .score-poor { border-color: #F43F5E; }
    .score-poor .score-number { color: #FB7185; }

    /* Progress Bars */
    .progress-wrap {
        margin: 0.75rem 0;
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.35rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #CBD5E1;
    }
    .progress-bar-bg {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 9999px;
        height: 7px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Skill Tags */
    .skill-section {
        margin: 0.5rem 0;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .skill-tag {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .skill-matched {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    .skill-missing {
        background: rgba(244, 63, 94, 0.12);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.28);
    }
    .skill-extra {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.28);
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green { background: rgba(16, 185, 129, 0.16); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-blue { background: rgba(59, 130, 246, 0.16); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-yellow { background: rgba(245, 158, 11, 0.16); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-red { background: rgba(244, 63, 94, 0.16); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.16); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }

    /* Suggestion Cards */
    .suggestion-item {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 3px solid #3B82F6;
        border-radius: 0 10px 10px 0;
        padding: 0.85rem 1.1rem;
        margin: 0.55rem 0;
        color: #CBD5E1;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* Form & Input Enhancements */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.8rem !important;
        cursor: pointer !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.95 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    }
    .stFileUploader {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px dashed rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
    }
    .divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        margin: 1.4rem 0;
    }
    .info-box {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        color: #93C5FD;
        font-size: 0.85rem;
        margin-top: 0.6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# UI Helper Functions
# ─────────────────────────────────────────────
def render_progress(label: str, value: float, color: str = "#3B82F6", max_val: float = 100):
    pct = min(100.0, max(0.0, (value / max_val) * 100))
    st.markdown(
        f"""
    <div class="progress-wrap">
        <div class="progress-label"><span>{label}</span><span>{value:.1f}%</span></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def score_style(score: float) -> tuple[str, str, str]:
    if score >= 80:
        return "score-excellent", "Excellent Match", "badge-green"
    elif score >= 60:
        return "score-good", "Strong Match", "badge-blue"
    elif score >= 40:
        return "score-average", "Moderate Match", "badge-yellow"
    else:
        return "score-poor", "Needs Calibration", "badge-red"


# ─────────────────────────────────────────────
# Header with Modern SVG Emblem Logo
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="brand-container">
    <div class="logo-badge">
        <span>ResCheck Verification</span>
        <span class="badge-tag">v2.0 NLP</span>
    </div>
    <div class="main-title-wrap">
        <svg width="42" height="42" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="48" height="48" rx="12" fill="url(#brandGrad)"/>
            <path d="M14 16C14 14.8954 14.8954 14 16 14H28L34 20V32C34 33.1046 33.1046 34 32 34H16C14.8954 34 14 33.1046 14 32V16Z" fill="#0F172A" fill-opacity="0.6"/>
            <path d="M28 14V20H34" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="31" cy="31" r="9" fill="#10B981"/>
            <path d="M28 31L30.2 33.2L34 29" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <defs>
                <linearGradient id="brandGrad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#2563EB"/>
                    <stop offset="1" stop-color="#4F46E5"/>
                </linearGradient>
            </defs>
        </svg>
        <h1 class="main-title">ResCheck</h1>
    </div>
    <div class="main-subtitle">
        Executive Resume ↔ Job Description Calibration & ATS Verification Engine
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Input Panels: Resume & Job Description
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="card-title">📄 Candidate Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF Resume",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        file_size_kb = uploaded_file.size / 1024
        st.markdown(
            f"""
        <div class="info-box">
            ✓ <strong>{uploaded_file.name}</strong> loaded ({file_size_kb:.1f} KB)
        </div>
        """,
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown('<div class="card-title">📋 Target Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description Text",
        height=190,
        placeholder="Paste target job specification, core qualifications, responsibilities, and required competencies...",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Centered Trigger Action Button
_, btn_col, _ = st.columns([1.6, 1.2, 1.6])
with btn_col:
    analyze_btn = st.button("Run ResCheck Analysis", use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Execution & Results Dashboard
# ─────────────────────────────────────────────
if analyze_btn:
    if not uploaded_file:
        st.error("Please provide a PDF resume to analyze.")
    elif not job_description.strip():
        st.error("Please provide a target job description.")
    else:
        with st.spinner("Analyzing document semantics, skills, and ATS alignment..."):
            try:
                t_start = time.perf_counter()
                parser = get_parser()
                matcher = get_matcher()
                scorer = get_scorer()

                # Step 1: Text extraction & normalization
                resume_raw = parser.extract_text_from_pdf(uploaded_file)
                resume_clean = parser.clean_text(resume_raw)
                jd_clean = parser.clean_text(job_description)

                # Step 2: NLP matching
                match_data = matcher.analyze(resume_clean, jd_clean)

                # Step 3: Scoring & synthesis
                score_data = scorer.compute_score(resume_clean, jd_clean, match_data)
                final_score = score_data["final_score"]
                sc_class, verdict, verdict_badge = score_style(final_score)
                latency_ms = (time.perf_counter() - t_start) * 1000
                resume_words = len(resume_clean.split())
                jd_words = len(jd_clean.split())

                st.success("Verification complete.")
                
                # Real-Time Telemetry Bar (Live Measured Metrics)
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; background:rgba(30,41,59,0.5); padding:0.45rem 1rem; border-radius:8px; font-size:0.8rem; color:#94A3B8; margin-bottom:1.2rem; border:1px solid rgba(255,255,255,0.06);'>
                        <span>⚡ Latency: <strong style='color:#F1F5F9;'>{latency_ms:.1f} ms</strong></span>
                        <span>Model: <strong style='color:#60A5FA;'>all-MiniLM-L6-v2</strong></span>
                        <span>Document Scope: <strong style='color:#CBD5E1;'>{resume_words} words (Resume) / {jd_words} words (JD)</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 1: Overall Match Score & Breakdown ──
                r1_col1, r1_col2 = st.columns([1, 2], gap="large")

                with r1_col1:
                    st.markdown(
                        f"""
                    <div class="card">
                        <div class="card-title">Match Calibration</div>
                        <div class="score-container">
                            <div class="score-circle {sc_class}">
                                <span class="score-number">{final_score:.0f}</span>
                                <span class="score-label">out of 100</span>
                            </div>
                            <span class="badge {verdict_badge}">{verdict}</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with r1_col2:
                    st.markdown(
                        '<div class="card"><div class="card-title">Score Dimensions</div>',
                        unsafe_allow_html=True,
                    )
                    breakdown = score_data["breakdown"]
                    render_progress("Skills Competency (40%)", breakdown["skills_score"], "#10B981")
                    render_progress("Experience Alignment (30%)", breakdown["experience_score"], "#3B82F6")
                    render_progress("ATS Keyword Alignment (20%)", breakdown["keyword_score"], "#8B5CF6")
                    render_progress("Education Prerequisites (10%)", breakdown["education_score"], "#F59E0B")
                    render_progress("Semantic Context Match", breakdown["semantic_similarity"] * 100, "#0EA5E9")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 2: Skills Analysis ──
                st.markdown(
                    '<div class="card"><div class="card-title">Competency Evaluation</div>',
                    unsafe_allow_html=True,
                )
                sk_col1, sk_col2, sk_col3 = st.columns(3)

                with sk_col1:
                    matched = match_data.get("matched_skills", [])
                    st.markdown(
                        f"**Matched Competencies** <span class='badge badge-green'>{len(matched)}</span>",
                        unsafe_allow_html=True,
                    )
                    if matched:
                        tags = "".join([f"<span class='skill-tag skill-matched'>{s}</span>" for s in matched])
                        st.markdown(f"<div class='skill-section'>{tags}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("No direct skill matches detected.")

                with sk_col2:
                    missing = match_data.get("missing_skills", [])
                    st.markdown(
                        f"**Missing Qualifications** <span class='badge badge-red'>{len(missing)}</span>",
                        unsafe_allow_html=True,
                    )
                    if missing:
                        tags = "".join([f"<span class='skill-tag skill-missing'>{s}</span>" for s in missing])
                        st.markdown(f"<div class='skill-section'>{tags}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("All primary requirements represented.")

                with sk_col3:
                    extra = match_data.get("extra_skills", [])
                    st.markdown(
                        f"**Additional Resume Skills** <span class='badge badge-blue'>{len(extra)}</span>",
                        unsafe_allow_html=True,
                    )
                    if extra:
                        tags = "".join([f"<span class='skill-tag skill-extra'>{s}</span>" for s in extra[:12]])
                        st.markdown(f"<div class='skill-section'>{tags}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("No peripheral skills listed.")

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 3: ATS Keywords & Experience/Education ──
                kw_col1, kw_col2 = st.columns(2, gap="large")

                with kw_col1:
                    st.markdown(
                        '<div class="card"><div class="card-title">ATS Keyword Verification</div>',
                        unsafe_allow_html=True,
                    )
                    kw_matched = match_data.get("ats_keywords_matched", [])
                    kw_missing = match_data.get("ats_keywords_missing", [])
                    total_ats = len(kw_matched) + len(kw_missing)
                    ats_rate = (len(kw_matched) / max(total_ats, 1)) * 100
                    render_progress("ATS Keyword Saturation", ats_rate, "#8B5CF6")

                    st.markdown(
                        f"""
                    <div style='margin-top:0.8rem;'>
                        <strong style='color:#CBD5E1; font-size:0.82rem;'>Found in Candidate Resume:</strong><br>
                        {"".join([f"<span class='skill-tag skill-matched' style='margin-top:0.25rem'>{k}</span>" for k in kw_matched[:10]]) or "<span style='color:#64748B'>None detected</span>"}
                    </div>
                    <div style='margin-top:0.8rem;'>
                        <strong style='color:#CBD5E1; font-size:0.82rem;'>Missing ATS Keywords:</strong><br>
                        {"".join([f"<span class='skill-tag skill-missing' style='margin-top:0.25rem'>{k}</span>" for k in kw_missing[:10]]) or "<span style='color:#64748B'>No gaps</span>"}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with kw_col2:
                    st.markdown(
                        '<div class="card"><div class="card-title">Background & Credentials</div>',
                        unsafe_allow_html=True,
                    )
                    exp = match_data.get("experience_analysis", {})
                    edu = match_data.get("education_analysis", {})

                    st.markdown(
                        f"""
                    <div style='margin-bottom:0.9rem;'>
                        <div style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.35rem;'>Professional Experience</div>
                        <div style='color:#F1F5F9; font-size:0.88rem;'>{exp.get("summary", "Experience tenure could not be calibrated.")}</div>
                        <div style='margin-top:0.4rem;'>
                            {"".join([f"<span class='badge badge-blue' style='margin:0.12rem'>{y}</span>" for y in exp.get("years_found", [])])}
                        </div>
                    </div>
                    <hr class='divider'>
                    <div>
                        <div style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.35rem;'>Educational Credentials</div>
                        <div style='color:#F1F5F9; font-size:0.88rem;'>{edu.get("summary", "No degree levels detected.")}</div>
                        <div style='margin-top:0.4rem;'>
                            {"".join([f"<span class='badge badge-purple' style='margin:0.12rem'>{d}</span>" for d in edu.get("degrees_found", [])])}
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 4: Actionable Recommendations ──
                st.markdown(
                    '<div class="card"><div class="card-title">Targeted Optimization Recommendations</div>',
                    unsafe_allow_html=True,
                )
                suggestions = score_data.get("suggestions", [])
                if suggestions:
                    s_col1, s_col2 = st.columns(2)
                    midpoint = len(suggestions) // 2 + len(suggestions) % 2
                    with s_col1:
                        for s in suggestions[:midpoint]:
                            st.markdown(f"<div class='suggestion-item'>⚡ {s}</div>", unsafe_allow_html=True)
                    with s_col2:
                        for s in suggestions[midpoint:]:
                            st.markdown(f"<div class='suggestion-item'>⚡ {s}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='suggestion-item'>✓ Resume demonstrates exceptional calibration with JD criteria.</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Raw Parsed Text Expander ──
                with st.expander("Inspected Cleaned Resume Stream"):
                    st.text_area("Extracted Text", resume_clean, height=180, disabled=True)

                # ── Production Telemetry & Diagnostics ──
                with st.expander("📊 Analysis Diagnostics & Telemetry"):
                    st.markdown("""
                    <div style='margin-bottom:0.8rem; font-size:0.85rem; color:#94A3B8;'>
                        Live diagnostic telemetry captured during this specific document evaluation:
                    </div>
                    """, unsafe_allow_html=True)
                    
                    jd_skills_detected = len(match_data.get("jd_skills", []))
                    confidence_label = "High" if jd_skills_detected >= 4 else ("Moderate" if jd_skills_detected >= 1 else "Low (Few JD Skills Detected)")

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Execution Latency", f"{latency_ms:.1f} ms")
                    m_col2.metric("JD Skills Detected", f"{jd_skills_detected} skills")
                    m_col3.metric("Candidate Skills", f"{len(match_data.get('resume_skills', []))} skills")
                    m_col4.metric("JD Signal Strength", confidence_label)

            except Exception as exc:
                st.error(f"Analysis encountered an issue: {str(exc)}")
                st.exception(exc)

elif not analyze_btn:
    st.markdown(
        """
    <div style='text-align:center; padding:2.5rem 0; color:#64748B;'>
        <div style='font-size:2rem; margin-bottom:0.8rem; opacity:0.8;'>📁</div>
        <div style='font-family:Plus Jakarta Sans,sans-serif; font-size:1.05rem; font-weight:600; color:#94A3B8;'>
            Upload candidate resume and paste job description to initiate calibration
        </div>
        <div style='font-size:0.82rem; margin-top:0.4rem; color:#64748B;'>
            Verifies skill coverage, tenure criteria, degree credentials, and ATS compliance
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
