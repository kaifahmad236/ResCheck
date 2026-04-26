"""
app.py - Main Streamlit UI for AI-Based Resume–Job Description Matching System
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.parser import ResumeParser
from backend.matcher import ResumeMatcher
from backend.scorer import ResumeScorer

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f0a 100%);
        min-height: 100vh;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff88 0%, #00d4ff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0;
    }
    .main-subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        font-weight: 300;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #e5e7eb;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Score circle */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
    }
    .score-circle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        position: relative;
    }
    .score-number {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.7;
        margin-top: 0.2rem;
    }

    /* Score colors */
    .score-excellent { border: 4px solid #00ff88; box-shadow: 0 0 40px rgba(0,255,136,0.3); }
    .score-excellent .score-number { color: #00ff88; }
    .score-good { border: 4px solid #00d4ff; box-shadow: 0 0 40px rgba(0,212,255,0.3); }
    .score-good .score-number { color: #00d4ff; }
    .score-average { border: 4px solid #fbbf24; box-shadow: 0 0 40px rgba(251,191,36,0.3); }
    .score-average .score-number { color: #fbbf24; }
    .score-poor { border: 4px solid #f87171; box-shadow: 0 0 40px rgba(248,113,113,0.3); }
    .score-poor .score-number { color: #f87171; }

    /* Progress bars */
    .progress-wrap {
        margin: 0.6rem 0;
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
        color: #9ca3af;
    }
    .progress-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 1s ease;
    }

    /* Skill tags */
    .skill-section {
        margin: 0.5rem 0;
    }
    .skill-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    .skill-matched {
        background: rgba(0,255,136,0.12);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    .skill-missing {
        background: rgba(248,113,113,0.12);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.3);
    }
    .skill-extra {
        background: rgba(0,212,255,0.12);
        color: #00d4ff;
        border: 1px solid rgba(0,212,255,0.3);
    }

    /* Suggestion cards */
    .suggestion-item {
        background: rgba(168,85,247,0.06);
        border-left: 3px solid #a855f7;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #d1d5db;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Keyword highlight */
    .keyword-matched {
        background: rgba(0,255,136,0.15);
        color: #00ff88;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 500;
    }
    .keyword-missing {
        background: rgba(248,113,113,0.15);
        color: #f87171;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 500;
        text-decoration: line-through;
        text-decoration-color: rgba(248,113,113,0.5);
    }

    /* Section divider */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-green { background: rgba(0,255,136,0.15); color: #00ff88; }
    .badge-red { background: rgba(248,113,113,0.15); color: #f87171; }
    .badge-blue { background: rgba(0,212,255,0.15); color: #00d4ff; }
    .badge-purple { background: rgba(168,85,247,0.15); color: #a855f7; }
    .badge-yellow { background: rgba(251,191,36,0.15); color: #fbbf24; }

    /* Streamlit overrides */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(0,255,136,0.4) !important;
        box-shadow: 0 0 0 2px rgba(0,255,136,0.1) !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%) !important;
        color: #000 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.05em !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(0,255,136,0.3) !important;
    }
    .stFileUploader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }
    .uploadedFileName {
        color: #00ff88 !important;
    }
    h1, h2, h3 { color: #f9fafb !important; }
    p, li { color: #9ca3af; }
    .stAlert { border-radius: 10px !important; }

    /* Info boxes */
    .info-box {
        background: rgba(0,212,255,0.06);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #7dd3fc;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #9ca3af !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: Render progress bar
# ─────────────────────────────────────────────
def render_progress(label: str, value: float, color: str = "#00ff88", max_val: float = 100):
    pct = min(100, (value / max_val) * 100)
    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-label"><span>{label}</span><span>{value:.1f}%</span></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def score_class(score: float) -> str:
    if score >= 80:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-average"
    else:
        return "score-poor"


def score_verdict(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Excellent Match", "badge-green"
    elif score >= 60:
        return "Good Match", "badge-blue"
    elif score >= 40:
        return "Moderate Match", "badge-yellow"
    else:
        return "Weak Match", "badge-red"


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🎯 ResumeMatch AI</div>
    <div class="main-subtitle">Intelligent Resume ↔ Job Description Analysis Powered by NLP</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="card-title">📄 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF resume here",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.markdown(f"""
        <div class="info-box">
            ✅ <strong>{uploaded_file.name}</strong> uploaded — {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card-title">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the full job description here",
        height=200,
        placeholder="Paste the job description here...\n\nInclude requirements, responsibilities, and required skills for the best analysis.",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Center the analyze button
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    analyze_btn = st.button("⚡ Analyze Match", use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Analysis & Results
# ─────────────────────────────────────────────
if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF resume before analyzing.")
    elif not job_description.strip():
        st.error("⚠️ Please paste a job description before analyzing.")
    else:
        with st.spinner("🔍 Parsing resume and analyzing match..."):
            try:
                # 1. Parse resume
                parser = ResumeParser()
                resume_text = parser.extract_text_from_pdf(uploaded_file)
                resume_clean = parser.clean_text(resume_text)

                # 2. Process JD
                jd_clean = parser.clean_text(job_description)

                # 3. Match
                matcher = ResumeMatcher()
                match_data = matcher.analyze(resume_clean, jd_clean)

                # 4. Score
                scorer = ResumeScorer()
                score_data = scorer.compute_score(resume_clean, jd_clean, match_data)

                final_score = score_data["final_score"]
                verdict, verdict_badge = score_verdict(final_score)
                sc = score_class(final_score)

                st.success("✅ Analysis complete!")
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 1: Score + Breakdown ──────────────────────
                r1_col1, r1_col2 = st.columns([1, 2], gap="large")

                with r1_col1:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">🏆 Overall Score</div>
                        <div class="score-container">
                            <div class="score-circle {sc}">
                                <span class="score-number">{final_score:.0f}</span>
                                <span class="score-label">out of 100</span>
                            </div>
                            <span class="badge {verdict_badge}">{verdict}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with r1_col2:
                    st.markdown('<div class="card"><div class="card-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
                    breakdown = score_data["breakdown"]
                    render_progress("Skills Match (40%)", breakdown["skills_score"], "#00ff88")
                    render_progress("Experience Alignment (30%)", breakdown["experience_score"], "#00d4ff")
                    render_progress("ATS Keywords (20%)", breakdown["keyword_score"], "#a855f7")
                    render_progress("Education Match (10%)", breakdown["education_score"], "#fbbf24")
                    render_progress("Semantic Similarity", breakdown["semantic_similarity"] * 100, "#fb923c")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 2: Skills Analysis ────────────────────────
                st.markdown('<div class="card"><div class="card-title">🛠️ Skills Analysis</div>', unsafe_allow_html=True)
                sk_col1, sk_col2, sk_col3 = st.columns(3)

                with sk_col1:
                    matched = match_data.get("matched_skills", [])
                    st.markdown(f"**✅ Matched Skills** <span class='badge badge-green'>{len(matched)}</span>", unsafe_allow_html=True)
                    if matched:
                        tags = "".join([f"<span class='skill-tag skill-matched'>{s}</span>" for s in matched])
                        st.markdown(f"<div class='skill-section'>{tags}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("No matched skills found.")

                with sk_col2:
                    missing = match_data.get("missing_skills", [])
                    st.markdown(f"**❌ Missing Skills** <span class='badge badge-red'>{len(missing)}</span>", unsafe_allow_html=True)
                    if missing:
                        tags = "".join([f"<span class='skill-tag skill-missing'>{s}</span>" for s in missing])
                        st.markdown(f"<div class='skill-section'>{tags}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("No critical missing skills!")

                with sk_col3:
                    extra = match_data.get("extra_skills", [])
                    st.markdown(f"**➕ Extra Skills** <span class='badge badge-blue'>{len(extra)}</span>", unsafe_allow_html=True)
                    if extra:
                        tags = "".join([f"<span class='skill-tag skill-extra'>{s}</span>" for s in extra[:15]])
                        st.markdown(f"<div class='skill-section'>{tags}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("No additional skills found.")

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 3: ATS Keywords ───────────────────────────
                kw_col1, kw_col2 = st.columns(2, gap="large")

                with kw_col1:
                    st.markdown('<div class="card"><div class="card-title">🔍 ATS Keyword Analysis</div>', unsafe_allow_html=True)
                    kw_matched = match_data.get("ats_keywords_matched", [])
                    kw_missing = match_data.get("ats_keywords_missing", [])
                    ats_rate = (len(kw_matched) / max(len(kw_matched) + len(kw_missing), 1)) * 100
                    render_progress("ATS Coverage", ats_rate, "#a855f7")
                    st.markdown(f"""
                    <div style='margin-top:0.8rem;'>
                        <strong style='color:#e5e7eb; font-size:0.85rem;'>✅ Found in Resume:</strong><br>
                        {"".join([f"<span class='skill-tag skill-matched' style='margin-top:0.3rem'>{k}</span>" for k in kw_matched[:12]]) or "<span style='color:#6b7280'>None detected</span>"}
                    </div>
                    <div style='margin-top:0.8rem;'>
                        <strong style='color:#e5e7eb; font-size:0.85rem;'>❌ Missing ATS Keywords:</strong><br>
                        {"".join([f"<span class='skill-tag skill-missing' style='margin-top:0.3rem'>{k}</span>" for k in kw_missing[:12]]) or "<span style='color:#6b7280'>None missing</span>"}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with kw_col2:
                    st.markdown('<div class="card"><div class="card-title">🎓 Experience & Education</div>', unsafe_allow_html=True)
                    exp = match_data.get("experience_analysis", {})
                    edu = match_data.get("education_analysis", {})

                    st.markdown(f"""
                    <div style='margin-bottom:1rem;'>
                        <div style='color:#9ca3af; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Experience</div>
                        <div style='color:#e5e7eb; font-size:0.9rem;'>{exp.get("summary", "Unable to determine experience level.")}</div>
                        <div style='margin-top:0.5rem;'>
                            {"".join([f"<span class='badge badge-blue' style='margin:0.15rem'>{y}</span>" for y in exp.get("years_found", [])])}
                        </div>
                    </div>
                    <hr class='divider'>
                    <div>
                        <div style='color:#9ca3af; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Education</div>
                        <div style='color:#e5e7eb; font-size:0.9rem;'>{edu.get("summary", "Unable to determine education level.")}</div>
                        <div style='margin-top:0.5rem;'>
                            {"".join([f"<span class='badge badge-purple' style='margin:0.15rem'>{d}</span>" for d in edu.get("degrees_found", [])])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Row 4: AI Suggestions ────────────────────────
                st.markdown('<div class="card"><div class="card-title">🤖 AI Improvement Suggestions</div>', unsafe_allow_html=True)
                suggestions = score_data.get("suggestions", [])
                if suggestions:
                    s_col1, s_col2 = st.columns(2)
                    half = len(suggestions) // 2 + len(suggestions) % 2
                    with s_col1:
                        for s in suggestions[:half]:
                            st.markdown(f"<div class='suggestion-item'>💡 {s}</div>", unsafe_allow_html=True)
                    with s_col2:
                        for s in suggestions[half:]:
                            st.markdown(f"<div class='suggestion-item'>💡 {s}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='suggestion-item'>✨ Your resume is well-optimized for this job description!</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Expandable Raw Text ──────────────────────────
                with st.expander("🔎 View Extracted Resume Text"):
                    st.text_area("Parsed Resume", resume_clean, height=200, disabled=True)

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.exception(e)

elif not analyze_btn:
    # Instructions when nothing has been analyzed yet
    st.markdown("""
    <div style='text-align:center; padding:2rem 0; color:#4b5563;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>⬆️</div>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:600; color:#6b7280;'>
            Upload your resume and paste a job description above to get started
        </div>
        <div style='font-size:0.85rem; margin-top:0.5rem; color:#4b5563;'>
            Analyzes skills, keywords, experience, and provides ATS optimization tips
        </div>
    </div>
    """, unsafe_allow_html=True)