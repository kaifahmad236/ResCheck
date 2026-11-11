"""
backend/scorer.py
─────────────────
Computes the weighted match score for ResCheck and generates actionable
optimization suggestions for applicants and recruiters.

Score weight configuration:
  - Skills Match       : 40%
  - Experience Match   : 30%
  - ATS Keyword Match  : 20%
  - Education Match    : 10%
  - Semantic Similarity: Bonus modifier (up to +5%)
"""

import re
import logging
from typing import Dict, List
from backend.matcher import DEGREE_HIERARCHY

logger = logging.getLogger(__name__)


# Standardized Scoring Weights
WEIGHTS: Dict[str, float] = {
    "skills": 0.40,
    "experience": 0.30,
    "keywords": 0.20,
    "education": 0.10,
}


class ResumeScorer:
    """
    Evaluates analysis output from ResumeMatcher, synthesizes sub-scores into
    a unified composite score (0-100), and outputs targeted recommendations.
    """

    def compute_score(
        self,
        resume_text: str,
        jd_text: str,
        match_data: dict,
    ) -> dict:
        """Computes sub-scores and synthesized weighted match score."""
        breakdown = {
            "skills_score": self._compute_skills_score(match_data),
            "experience_score": self._compute_experience_score(match_data),
            "keyword_score": self._compute_keyword_score(match_data),
            "education_score": self._compute_education_score(match_data),
        }

        semantic_sim = float(match_data.get("semantic_similarity", 0.0))
        breakdown["semantic_similarity"] = semantic_sim

        # Weighted aggregate
        raw_score = (
            breakdown["skills_score"] * WEIGHTS["skills"]
            + breakdown["experience_score"] * WEIGHTS["experience"]
            + breakdown["keyword_score"] * WEIGHTS["keywords"]
            + breakdown["education_score"] * WEIGHTS["education"]
        )

        # Contextual semantic bonus (up to 5 percentage points)
        semantic_bonus = semantic_sim * 5.0
        final_score = min(100.0, raw_score + semantic_bonus)

        suggestions = self._generate_suggestions(
            resume_text=resume_text,
            match_data=match_data,
            breakdown=breakdown,
            final_score=final_score,
        )

        return {
            "final_score": round(final_score, 1),
            "breakdown": breakdown,
            "suggestions": suggestions,
            "weights": WEIGHTS,
        }

    def _compute_skills_score(self, match_data: dict) -> float:
        """Evaluates skill overlap ratio with adaptive scaling."""
        matched = len(match_data.get("matched_skills", []))
        missing = len(match_data.get("missing_skills", []))
        total_jd = matched + missing

        if total_jd == 0:
            return float(match_data.get("semantic_similarity", 0.0)) * 75.0

        base_ratio = matched / total_jd
        score = base_ratio * 100.0

        # Gentle penalty for high volume of missing competencies (capped at 15 points)
        penalty = min(missing * 1.5, 15.0)
        # Small boost for each verified match
        boost = min(matched * 1.0, 10.0)

        adjusted = score - penalty + boost
        return round(float(max(0.0, min(100.0, adjusted))), 1)

    def _compute_experience_score(self, match_data: dict) -> float:
        """Scores candidate experience relative to JD expectations."""
        exp = match_data.get("experience_analysis", {})
        required = exp.get("required_years")
        actual = exp.get("actual_years")

        if required is None and actual is None:
            return 70.0  # Neutral baseline when neither document specifies years

        if required is None:
            return 85.0 if actual else 70.0

        if actual is None:
            return 50.0  # Requirement stated but no verifiable years detected

        if actual >= required:
            # Reward meeting or exceeding experience
            excess = min((actual - required) / max(required, 1), 1.0)
            return round(min(100.0, 85.0 + (excess * 15.0)), 1)
        else:
            ratio = actual / required
            return round(max(0.0, ratio * 80.0), 1)

    def _compute_keyword_score(self, match_data: dict) -> float:
        """ATS keyword alignment score based on critical domain tokens."""
        matched = match_data.get("ats_keywords_matched", [])
        missing = match_data.get("ats_keywords_missing", [])
        total = len(matched) + len(missing)

        if total == 0:
            return 65.0

        return round((len(matched) / total) * 100.0, 1)

    def _compute_education_score(self, match_data: dict) -> float:
        """Evaluates formal educational degree alignment."""
        edu = match_data.get("education_analysis", {})
        resume_degrees = edu.get("resume_degrees", [])
        jd_degrees = edu.get("jd_degrees", [])

        if not jd_degrees:
            return 100.0 if resume_degrees else 80.0

        if not resume_degrees:
            return 45.0

        resume_max = max((DEGREE_HIERARCHY.get(d, 0) for d in resume_degrees), default=0)
        jd_min = min((DEGREE_HIERARCHY.get(d, 0) for d in jd_degrees), default=0)

        if resume_max >= jd_min:
            return 100.0
        elif resume_max == jd_min - 1:
            return 75.0
        else:
            return 45.0

    def _generate_suggestions(
        self,
        resume_text: str,
        match_data: dict,
        breakdown: dict,
        final_score: float,
    ) -> List[str]:
        """Generates prioritized, high-impact recommendations."""
        suggestions = []

        # 1. Missing Core Skills
        missing_skills = match_data.get("missing_skills", [])
        if missing_skills:
            top_skills = missing_skills[:5]
            suggestions.append(
                f"Integrate key missing skills: {', '.join(repr(s) for s in top_skills)}. "
                "These are explicitly requested by the employer."
            )

        # 2. ATS Keyword Coverage
        missing_ats = match_data.get("ats_keywords_missing", [])
        if missing_ats:
            top_ats = missing_ats[:4]
            suggestions.append(
                f"Optimize for ATS scanning by including prominent keywords: {', '.join(repr(k) for k in top_ats)} "
                "within your experience bullet points."
            )

        # 3. Experience Gap or Calibration
        exp = match_data.get("experience_analysis", {})
        required_yrs = exp.get("required_years")
        actual_yrs = exp.get("actual_years")

        if required_yrs and actual_yrs and actual_yrs < required_yrs:
            diff = required_yrs - actual_yrs
            suggestions.append(
                f"Experience Gap: Position requires {required_yrs}+ years (detected ~{actual_yrs} years). "
                f"Highlight relevant project scale, freelance roles, or certifications to bridge the {diff}-year difference."
            )
        elif breakdown["experience_score"] < 65:
            suggestions.append(
                "Strengthen your experience section with clear impact statements and dates (e.g., '2021 – Present')."
            )

        # 4. Metric & Quantification Check
        resume_lower = resume_text.lower()
        has_metrics = bool(re.search(r"\d+%|\$\d+|\d+\s*(?:k|m|million|thousand|users|clients|engineers)", resume_lower))
        if not has_metrics:
            suggestions.append(
                "Quantify your accomplishments. Use specific data points, performance improvements (e.g., 'reduced latency by 35%'), and team sizes."
            )

        # 5. Professional Summary Check
        has_summary = any(k in resume_lower for k in ["summary", "profile", "about me", "objective"])
        if not has_summary:
            suggestions.append(
                "Add an executive 3-line summary at the top of your resume highlighting your primary value proposition for this target role."
            )

        # 6. Portfolio / Technical Links
        has_portfolio = any(k in resume_lower for k in ["github", "linkedin", "portfolio", "gitlab"])
        if not has_portfolio:
            suggestions.append(
                "Include direct links to your GitHub profile, LinkedIn, or portfolio to validate your technical depth."
            )

        # 7. General formatting guidance
        if final_score < 75:
            suggestions.append(
                "Maintain a clean, ATS-compliant single-column layout without tables, graphics, or nested text boxes."
            )

        return suggestions[:7]
