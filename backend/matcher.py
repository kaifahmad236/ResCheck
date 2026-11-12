"""
backend/matcher.py
──────────────────
Core NLP matching engine for ResCheck.
- Skill extraction using precise boundary matching (prevents false substring positives)
- ATS keyword coverage analysis
- Experience duration and education level detection
- Semantic text similarity via sentence-transformers (with TF-IDF fallback)
"""

import re
import datetime
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from utils.text_cleaner import tokenize, extract_ngrams

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Master Skills & Domain Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

SKILLS_DB: Set[str] = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "golang",
    "ruby", "php", "swift", "kotlin", "rust", "scala", "r", "matlab", "perl",
    "shell", "bash", "powershell", "groovy", "dart", "lua",

    # Web Frontend
    "html", "css", "react", "reactjs", "react.js", "angular", "angularjs",
    "vue", "vue.js", "vuejs", "next.js", "nextjs", "nuxt", "svelte",
    "jquery", "bootstrap", "tailwind", "tailwindcss", "sass", "less",
    "webpack", "vite", "babel", "redux", "graphql", "rest", "restful",

    # Web Backend
    "node.js", "nodejs", "express", "fastapi", "flask", "django", "spring",
    "spring boot", "laravel", "rails", "asp.net", "gin", "fiber",
    "microservices", "api", "grpc", "websockets",

    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "cassandra", "oracle", "sqlite", "dynamodb", "firebase", "neo4j",
    "mariadb", "mssql", "nosql", "hbase", "couchdb", "influxdb",

    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "heroku", "docker", "kubernetes",
    "k8s", "terraform", "ansible", "jenkins", "github actions", "circleci",
    "travis ci", "gitlab ci", "ci/cd", "devops", "helm", "istio",
    "cloudformation", "pulumi", "prometheus", "grafana", "datadog",

    # Data Science & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "xgboost", "lightgbm", "huggingface", "transformers", "bert", "gpt",
    "llm", "rag", "langchain", "openai", "data science", "data analysis",
    "feature engineering", "model deployment", "mlops", "mlflow",
    "airflow", "spark", "hadoop", "hive", "kafka", "flink",

    # Tools & Platforms
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
    "linux", "unix", "windows", "macos", "vim", "vs code", "intellij",
    "postman", "swagger", "figma", "adobe", "photoshop",

    # Testing
    "unit testing", "integration testing", "selenium", "pytest", "jest",
    "mocha", "cypress", "playwright", "junit", "testng",

    # Methodologies & Architecture
    "agile", "scrum", "kanban", "tdd", "bdd", "oop", "solid",
    "design patterns", "serverless", "event-driven",

    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",

    # Security
    "cybersecurity", "penetration testing", "oauth", "jwt", "ssl", "tls",
    "encryption", "authentication", "authorization",

    # Soft skills & Leadership
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "project management", "time management",
}

# Precompile skill regex patterns to ensure high performance and strict boundaries
_SKILL_PATTERNS: Dict[str, re.Pattern] = {}
for _skill in SKILLS_DB:
    # Single-letter languages (R, C) require explicit programming context to avoid false positives
    if _skill in {"r", "c"}:
        _pattern = r"(?:\b(?:programming in|language|coding in)\s+)?\b" + re.escape(_skill) + r"\b(?:\s+(?:programming|language|developer))?"
    elif any(ch in _skill for ch in "+#.-/"):
        _pattern = r"(?:\b|(?<=\s))" + re.escape(_skill) + r"(?:\b|(?=[\s,.;:]|$))"
    else:
        _pattern = r"\b" + re.escape(_skill) + r"\b"
    _SKILL_PATTERNS[_skill] = re.compile(_pattern, re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Education Degree Hierarchy & Experience Rules
# ──────────────────────────────────────────────────────────────────────────────

DEGREE_HIERARCHY: Dict[str, int] = {
    "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
    "master": 4, "masters": 4, "m.s.": 4, "msc": 4, "m.tech": 4, "mba": 4,
    "bachelor": 3, "bachelors": 3, "b.s.": 3, "b.tech": 3, "bsc": 3, "b.e.": 3, "b.com": 3,
    "associate": 2, "diploma": 1,
}

DEGREE_KEYWORDS: List[str] = list(DEGREE_HIERARCHY.keys()) + ["high school", "ged"]

EXPERIENCE_PATTERNS: List[str] = [
    r"(\d+)\+?\s*years?\s*of\s*(?:relevant\s*)?experience",
    r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:work\s*)?experience",
    r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\s*(?:of\s*)?experience",
    r"(\d+)\s*-\s*(\d+)\s*years?\s*(?:of\s*)?experience",
    r"minimum\s*(?:of\s*)?(\d+)\s*years?",
    r"at\s*least\s*(\d+)\s*years?",
]


# ──────────────────────────────────────────────────────────────────────────────
# ResumeMatcher Engine
# ──────────────────────────────────────────────────────────────────────────────

class ResumeMatcher:
    """
    NLP Matching Engine evaluating skill alignment, semantic similarity,
    ATS keyword presence, experience, and education levels.
    """

    _cached_model = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self):
        """Loads and caches the SentenceTransformer model to prevent redundant re-instantiations."""
        if ResumeMatcher._cached_model is not None:
            return ResumeMatcher._cached_model

        try:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            try:
                model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                model = SentenceTransformer(self.model_name)

            ResumeMatcher._cached_model = model
            logger.info("SentenceTransformer model successfully loaded.")
            return model
        except Exception as exc:
            logger.warning(f"Could not load SentenceTransformer ({exc}). Using TF-IDF fallback.")
            return None

    def analyze(self, resume_text: str, jd_text: str) -> Dict:
        """Runs complete matching analysis between resume and job description."""
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)

        matched_skills = sorted(resume_skills & jd_skills)
        missing_skills = sorted(jd_skills - resume_skills)
        extra_skills = sorted(resume_skills - jd_skills)

        semantic_similarity = self.compute_semantic_similarity(resume_text, jd_text)
        ats_matched, ats_missing = self.ats_keyword_analysis(resume_text, jd_text)
        experience_analysis = self.analyze_experience(resume_text, jd_text)
        education_analysis = self.analyze_education(resume_text, jd_text)

        return {
            "resume_skills": list(resume_skills),
            "jd_skills": list(jd_skills),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "semantic_similarity": semantic_similarity,
            "ats_keywords_matched": ats_matched,
            "ats_keywords_missing": ats_missing,
            "experience_analysis": experience_analysis,
            "education_analysis": education_analysis,
            "skills_match_ratio": len(matched_skills) / max(len(jd_skills), 1),
        }

    def extract_skills(self, text: str) -> Set[str]:
        """Extracts recognized skills using precompiled word-boundary regexes."""
        if not text:
            return set()

        text_lower = text.lower()
        found_skills = set()

        for skill, pattern in _SKILL_PATTERNS.items():
            if pattern.search(text_lower):
                found_skills.add(skill)

        return found_skills

    def compute_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculates semantic similarity via embeddings or TF-IDF vectorization."""
        if self.model is not None:
            return self._similarity_transformers(resume_text, jd_text)
        return self._similarity_tfidf(resume_text, jd_text)

    def _similarity_transformers(self, resume_text: str, jd_text: str) -> float:
        from sklearn.metrics.pairwise import cosine_similarity

        # Cap text length to prevent memory saturation
        max_chars = 5000
        resume_trunc = resume_text[:max_chars]
        jd_trunc = jd_text[:max_chars]

        embeddings = self.model.encode([resume_trunc, jd_trunc])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def _similarity_tfidf(self, resume_text: str, jd_text: str) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
            matrix = vectorizer.fit_transform([resume_text, jd_text])
            sim = cosine_similarity(matrix[0], matrix[1])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0

    def ats_keyword_analysis(self, resume_text: str, jd_text: str) -> Tuple[List[str], List[str]]:
        """
        Identifies critical ATS keywords from the JD and checks resume alignment.
        Extracts prominent technical nouns, acronyms, and recognized competencies.
        """
        jd_keywords = self._extract_important_keywords(jd_text)
        resume_lower = resume_text.lower()

        matched = []
        missing = []

        for kw in jd_keywords:
            pattern = re.compile(r"\b" + re.escape(kw.lower()) + r"\b")
            if pattern.search(resume_lower):
                matched.append(kw)
            else:
                missing.append(kw)

        matched = sorted(matched, key=len, reverse=True)
        missing = sorted(missing, key=len, reverse=True)
        return matched[:20], missing[:20]

    def _extract_important_keywords(self, text: str) -> List[str]:
        keywords = set()
        text_lower = text.lower()

        # 1. Recognized skills present in the JD
        for skill in SKILLS_DB:
            if _SKILL_PATTERNS[skill].search(text_lower):
                keywords.add(skill.title())

        # 2. Capitalized phrases (multi-word technical entities)
        cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        for phrase in cap_phrases:
            if 3 < len(phrase) < 45:
                keywords.add(phrase)

        # 3. Uppercase acronyms (e.g., API, AWS, CI/CD, SQL, SDK)
        acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
        excluded = {"THE", "AND", "FOR", "WITH", "ARE", "YOU", "CAN", "NOT", "ALL", "HAS", "OUR", "NEW", "PER"}
        for acr in acronyms:
            if acr not in excluded:
                keywords.add(acr)

        return sorted(keywords)[:30]

    def analyze_experience(self, resume_text: str, jd_text: str) -> Dict:
        """Calculates years of experience requested vs. actual experience detected."""
        jd_years = self._extract_experience_years(jd_text)
        resume_years = self._extract_experience_years(resume_text)
        estimated_years = self._estimate_experience_from_dates(resume_text)

        required_years = max(jd_years) if jd_years else None
        all_resume_years = resume_years + ([estimated_years] if estimated_years else [])
        actual_years = max(all_resume_years) if all_resume_years else None

        if required_years and actual_years:
            if actual_years >= required_years:
                summary = f"Meets experience criteria: ~{actual_years} years detected ({required_years}+ required)."
            else:
                gap = required_years - actual_years
                summary = f"Experience gap: ~{actual_years} years detected ({required_years}+ required, {gap} yr difference)."
        elif required_years:
            summary = f"Job requires {required_years}+ years of experience."
        elif actual_years:
            summary = f"Resume indicates ~{actual_years} years of professional experience."
        else:
            summary = "Experience duration not explicitly identified."

        return {
            "summary": summary,
            "required_years": required_years,
            "actual_years": actual_years,
            "years_found": [f"{y} yr" for y in sorted(set(jd_years + resume_years))],
        }

    def _extract_experience_years(self, text: str) -> List[int]:
        years = []
        text_lower = text.lower()
        for pattern in EXPERIENCE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    years.extend([int(m) for m in match if m.isdigit()])
                elif str(match).isdigit():
                    years.append(int(match))
        return [y for y in years if 0 < y <= 40]

    def _estimate_experience_from_dates(self, text: str) -> Optional[int]:
        """Calculates approximate professional tenure based on date ranges."""
        year_ranges = re.findall(
            r"\b(19\d{2}|20\d{2})\s*[-–—to]+\s*(19\d{2}|20\d{2}|present|current)\b",
            text.lower(),
        )
        current_year = datetime.datetime.now().year
        total = 0

        for start, end in year_ranges:
            try:
                s = int(start)
                e = current_year if end in ("present", "current") else int(end)
                if 1990 <= s <= current_year and s <= e:
                    total += (e - s)
            except ValueError:
                continue

        return total if 0 < total <= 40 else None

    def analyze_education(self, resume_text: str, jd_text: str) -> Dict:
        """Evaluates educational credentials against requirements."""
        resume_degrees = self._extract_degrees(resume_text)
        jd_degrees = self._extract_degrees(jd_text)

        resume_max = max((DEGREE_HIERARCHY.get(d, 0) for d in resume_degrees), default=0)
        jd_min = min((DEGREE_HIERARCHY.get(d, 0) for d in jd_degrees), default=0) if jd_degrees else 0

        if resume_degrees and jd_degrees:
            if resume_max >= jd_min:
                summary = "Educational qualifications satisfy job prerequisites."
            else:
                summary = "Highest detected degree may be below the preferred level in JD."
        elif resume_degrees:
            summary = f"Credentials identified: {', '.join(d.title() for d in resume_degrees[:3])}."
        elif jd_degrees:
            summary = f"Prerequisite degree: {', '.join(d.title() for d in jd_degrees[:2])}."
        else:
            summary = "No mandatory degree prerequisites identified."

        return {
            "summary": summary,
            "resume_degrees": resume_degrees,
            "jd_degrees": jd_degrees,
            "degrees_found": [d.title() for d in (resume_degrees + jd_degrees)[:5]],
        }

    def _extract_degrees(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for deg in DEGREE_KEYWORDS:
            pattern = re.compile(r"\b" + re.escape(deg) + r"\b")
            if pattern.search(text_lower):
                found.append(deg)
        return found
