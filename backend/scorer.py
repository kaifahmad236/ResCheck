"""
backend/scorer.py
─────────────────
Computes the final weighted match score and generates
AI-powered improvement suggestions and ATS tips.

Score weights:
  Skills Match   : 40%
  Experience     : 30%
  ATS Keywords   : 20%
  Education      : 10%
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Score Weight Configuration
# ──────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skills": 0.40,
    "experience": 0.30,
    "keywords": 0.20,
    "education": 0.10,
}


class ResumeScorer:
    """
    Computes a final weighted match score and generates
    actionable improvement suggestions.
    """

    def compute_score(
        self,
        resume_text: str,
        jd_text: str,
        match_data: dict,
    ) -> dict:
        """
        Compute all sub-scores and combine into a final weighted score.

        Args:
            resume_text: Cleaned resume text
            jd_text: Cleaned job description text
            match_data: Output from ResumeMatcher.analyze()

        Returns:
            Dictionary with final_score, breakdown, and suggestions.
        """
        breakdown = {}

        # ── 1. Skills Score (40%) ──────────────────────────────────────────
        breakdown["skills_score"] = self._compute_skills_score(match_data)

        # ── 2. Experience Score (30%) ──────────────────────────────────────
        breakdown["experience_score"] = self._compute_experience_score(match_data)

        # ── 3. Keyword/ATS Score (20%) ─────────────────────────────────────
        breakdown["keyword_score"] = self._compute_keyword_score(match_data)

        # ── 4. Education Score (10%) ───────────────────────────────────────
        breakdown["education_score"] = self._compute_education_score(match_data)

        # ── 5. Semantic Similarity (used as a bonus modifier) ─────────────
        semantic_sim = match_data.get("semantic_similarity", 0.0)
        breakdown["semantic_similarity"] = semantic_sim

        # ── Final Weighted Score ───────────────────────────────────────────
        raw_score = (
            breakdown["skills_score"]     * WEIGHTS["skills"]
            + breakdown["experience_score"] * WEIGHTS["experience"]
            + breakdown["keyword_score"]    * WEIGHTS["keywords"]
            + breakdown["education_score"]  * WEIGHTS["education"]
        )

        # Apply a small semantic bonus (up to +5 points for very high similarity)
        semantic_bonus = semantic_sim * 5.0
        final_score = min(100.0, raw_score + semantic_bonus)

        # ── Generate Suggestions ──────────────────────────────────────────
        suggestions = self._generate_suggestions(
            resume_text, jd_text, match_data, breakdown, final_score
        )

        return {
            "final_score": round(final_score, 1),
            "breakdown": breakdown,
            "suggestions": suggestions,
            "weights": WEIGHTS,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Sub-score Calculations
    # ──────────────────────────────────────────────────────────────────────

    def _compute_skills_score(self, match_data: dict) -> float:
        """
        Skills score based on the ratio of matched JD skills found in resume.
        Penalised slightly for a large number of missing skills.

        Returns: 0–100
        """
        matched = len(match_data.get("matched_skills", []))
        missing = len(match_data.get("missing_skills", []))
        total_jd = matched + missing

        if total_jd == 0:
            # No skills detected in JD – use semantic similarity as proxy
            return match_data.get("semantic_similarity", 0.0) * 80

        ratio = matched / total_jd
        # Apply a logarithmic boost so partial matches still score reasonably
        score = ratio * 100

        # Small penalty per missing skill (capped)
        penalty = min(missing * 2, 20)
        score = max(0.0, score - penalty + (matched * 1.5))

        return round(min(score, 100.0), 1)

    def _compute_experience_score(self, match_data: dict) -> float:
        """
        Experience score based on years comparison.

        Returns: 0–100
        """
        exp = match_data.get("experience_analysis", {})
        required = exp.get("required_years")
        actual = exp.get("actual_years")

        if required is None and actual is None:
            # Cannot determine – give neutral score, let semantic carry it
            return 60.0

        if required is None:
            # Job doesn't specify years – reward any experience detected
            return 80.0 if actual else 60.0

        if actual is None:
            # We know what's required but couldn't detect resume experience
            return 50.0

        if actual >= required:
            # Bonus for exceeding requirement
            excess_ratio = min((actual - required) / max(required, 1), 1.0)
            return round(min(100.0, 80.0 + excess_ratio * 20.0), 1)
        else:
            # Proportional score based on how close they are
            ratio = actual / required
            return round(max(0.0, ratio * 80.0), 1)

    def _compute_keyword_score(self, match_data: dict) -> float:
        """
        ATS keyword score: ratio of matched keywords to total JD keywords.

        Returns: 0–100
        """
        matched = match_data.get("ats_keywords_matched", [])
        missing = match_data.get("ats_keywords_missing", [])
        total = len(matched) + len(missing)

        if total == 0:
            return 60.0  # Neutral if no keywords detected

        ratio = len(matched) / total
        return round(ratio * 100, 1)

    def _compute_education_score(self, match_data: dict) -> float:
        """
        Education score based on degree level alignment.

        Returns: 0–100
        """
        edu = match_data.get("education_analysis", {})
        resume_degrees = edu.get("resume_degrees", [])
        jd_degrees = edu.get("jd_degrees", [])

        if not jd_degrees:
            # No requirement specified
            return 100.0 if resume_degrees else 75.0

        if not resume_degrees:
            # JD has requirements but resume shows none
            return 40.0

        degree_hierarchy = {
            "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
            "master": 4, "masters": 4, "m.s.": 4, "msc": 4, "m.tech": 4, "mba": 4,
            "bachelor": 3, "bachelors": 3, "b.s.": 3, "b.tech": 3, "bsc": 3,
            "b.e.": 3, "b.com": 3,
            "associate": 2, "diploma": 1,
        }

        resume_max = max((degree_hierarchy.get(d, 0) for d in resume_degrees), default=0)
        jd_min = min((degree_hierarchy.get(d, 0) for d in jd_degrees), default=0)

        if resume_max >= jd_min:
            return 100.0
        elif resume_max == jd_min - 1:
            return 70.0  # One level below – close but not exact
        else:
            return 40.0  # Significantly below

    # ──────────────────────────────────────────────────────────────────────
    # AI Suggestions Generator
    # ──────────────────────────────────────────────────────────────────────

    def _generate_suggestions(
        self,
        resume_text: str,
        jd_text: str,
        match_data: dict,
        breakdown: dict,
        final_score: float,
    ) -> list[str]:
        """
        Generate specific, actionable improvement suggestions based on analysis.

        Returns:
            List of suggestion strings.
        """
        suggestions = []

        # ── Skills suggestions ────────────────────────────────────────────
        missing_skills = match_data.get("missing_skills", [])
        if missing_skills:
            top_missing = missing_skills[:5]
            skills_str = ", ".join([f"'{s}'" for s in top_missing])
            suggestions.append(
                f"Add these high-priority missing skills to your resume: {skills_str}. "
                f"These appear in the job description but are absent from your resume."
            )

        if breakdown["skills_score"] < 50:
            suggestions.append(
                "Your skills alignment is below 50%. Consider adding a dedicated 'Technical Skills' "
                "or 'Core Competencies' section that lists skills verbatim from the job description."
            )

        # ── ATS keyword suggestions ───────────────────────────────────────
        ats_missing = match_data.get("ats_keywords_missing", [])
        if ats_missing:
            top_ats = ats_missing[:5]
            ats_str = ", ".join([f"'{k}'" for k in top_ats])
            suggestions.append(
                f"Include these ATS-critical keywords in your resume: {ats_str}. "
                "Many companies use Applicant Tracking Systems that filter by exact keyword matches."
            )

        if breakdown["keyword_score"] < 60:
            suggestions.append(
                "Your ATS keyword coverage is low. Mirror the exact phrasing from the job description "
                "in your bullet points and skills section — avoid paraphrasing technical terms."
            )

        # ── Experience suggestions ────────────────────────────────────────
        exp = match_data.get("experience_analysis", {})
        required_years = exp.get("required_years")
        actual_years = exp.get("actual_years")

        if required_years and actual_years and actual_years < required_years:
            gap = required_years - actual_years
            suggestions.append(
                f"The job requires {required_years}+ years of experience but your resume suggests ~{actual_years}. "
                f"Quantify your impact in each role and consider including freelance, "
                f"internship, or project experience to bridge the {gap}-year gap."
            )
        elif breakdown["experience_score"] < 60:
            suggestions.append(
                "Strengthen your experience section by adding measurable achievements "
                "(e.g., 'Improved system performance by 30%', 'Led a team of 5 engineers'). "
                "Use numbers and impact statements."
            )

        # ── Education suggestions ─────────────────────────────────────────
        if breakdown["education_score"] < 70:
            edu = match_data.get("education_analysis", {})
            jd_degrees = edu.get("jd_degrees", [])
            if jd_degrees:
                suggestions.append(
                    f"The job description mentions a {', '.join(jd_degrees[:2])} degree. "
                    "If you have relevant coursework, certifications, or equivalent experience, "
                    "make sure to highlight those prominently."
                )

        # ── Semantic similarity suggestions ───────────────────────────────
        semantic_sim = match_data.get("semantic_similarity", 0)
        if semantic_sim < 0.5:
            suggestions.append(
                "The overall language of your resume is quite different from the job description. "
                "Rewrite your summary/objective section using terminology from the JD to improve contextual relevance."
            )

        # ── Structure and formatting suggestions ─────────────────────────
        resume_lower = resume_text.lower()

        if "summary" not in resume_lower and "objective" not in resume_lower and "profile" not in resume_lower:
            suggestions.append(
                "Add a professional summary (3–4 lines) at the top of your resume. "
                "Tailor it specifically to this role using keywords from the job description."
            )

        if not re.search(r"\d+%|\d+\s+(?:million|thousand|users|customers|projects)", resume_lower):
            suggestions.append(
                "Quantify your achievements wherever possible. "
                "Numbers stand out to recruiters: use metrics like percentages, "
                "revenue impact, team size, or project scale."
            )

        if "github" not in resume_lower and "portfolio" not in resume_lower and "project" not in resume_lower:
            suggestions.append(
                "Include links to relevant projects, your GitHub profile, or a portfolio. "
                "This is especially important for technical roles."
            )

        # ── General ATS formatting tips ────────────────────────────────────
        if final_score < 70:
            suggestions.append(
                "ATS Tip: Use a clean, single-column resume format. "
                "Avoid tables, text boxes, headers/footers, and images — these confuse ATS parsers."
            )

        suggestions.append(
            "Tailor this resume specifically for each application. "
            "A targeted resume scores 40% higher on average than a generic one."
        )

        return suggestions[:10]  # Cap at 10 suggestions