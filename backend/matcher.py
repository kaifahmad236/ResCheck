"""
backend/matcher.py
──────────────────
Core NLP matching engine.
- Extracts skills from resume and JD
- Identifies matched, missing, and extra skills
- Analyzes ATS keywords
- Analyzes experience and education alignment
- Uses sentence-transformers for semantic similarity
"""

import re
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Master Skills Database
# ──────────────────────────────────────────────────────────────────────────────

SKILLS_DB = {
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

    # Methodologies
    "agile", "scrum", "kanban", "tdd", "bdd", "oop", "solid",
    "design patterns", "microservices", "serverless", "event-driven",

    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",

    # Security
    "cybersecurity", "penetration testing", "oauth", "jwt", "ssl", "tls",
    "encryption", "authentication", "authorization",

    # Soft skills (domain-specific)
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "project management", "time management",
}

# Degree keywords
DEGREE_KEYWORDS = [
    "phd", "ph.d", "doctorate", "doctoral",
    "master", "masters", "m.s.", "msc", "m.tech", "mba",
    "bachelor", "bachelors", "b.s.", "b.tech", "bsc", "b.e.", "b.com",
    "associate", "diploma", "high school", "ged",
]

# Experience patterns (years of experience)
EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s*of\s*(?:relevant\s*)?experience",
    r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:work\s*)?experience",
    r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\s*(?:of\s*)?experience",
    r"(\d+)\s*-\s*(\d+)\s*years?\s*(?:of\s*)?experience",
    r"minimum\s*(?:of\s*)?(\d+)\s*years?",
    r"at\s*least\s*(\d+)\s*years?",
]


# ──────────────────────────────────────────────────────────────────────────────
# ResumeMatcher Class
# ──────────────────────────────────────────────────────────────────────────────

class ResumeMatcher:
    """
    Core matching engine that:
    1. Extracts skills from resume and JD
    2. Computes semantic similarity using sentence-transformers
    3. Analyzes ATS keyword coverage
    4. Parses experience and education requirements
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load sentence-transformer model (lazy + cached)."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic similarity will use TF-IDF fallback."
            )
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    # ──────────────────────────────────────────────────────────────────────
    # Main Analysis Entry Point
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """
        Run full matching analysis between resume and job description.

        Args:
            resume_text: Cleaned resume text
            jd_text: Cleaned job description text

        Returns:
            Dictionary containing all match data.
        """
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
            "skills_match_ratio": (
                len(matched_skills) / max(len(jd_skills), 1)
            ),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Skill Extraction
    # ──────────────────────────────────────────────────────────────────────

    def extract_skills(self, text: str) -> set:
        """
        Extract skills from text using the master skills database.

        Uses word-boundary matching to avoid false positives
        (e.g., 'R' only matches as a standalone token).

        Args:
            text: Cleaned text to extract skills from

        Returns:
            Set of matched skill strings.
        """
        text_lower = text.lower()
        found_skills = set()

        for skill in SKILLS_DB:
            # Use word boundary for short skills to avoid partial matches
            if len(skill) <= 2:
                pattern = r"\b" + re.escape(skill) + r"\b"
            else:
                pattern = re.escape(skill)

            if re.search(pattern, text_lower):
                found_skills.add(skill)

        return found_skills

    # ──────────────────────────────────────────────────────────────────────
    # Semantic Similarity
    # ──────────────────────────────────────────────────────────────────────

    def compute_semantic_similarity(
        self, resume_text: str, jd_text: str
    ) -> float:
        """
        Compute cosine similarity between resume and JD embeddings.

        Uses sentence-transformers if available, falls back to TF-IDF.

        Returns:
            Float between 0.0 and 1.0.
        """
        if self.model is not None:
            return self._semantic_similarity_transformers(resume_text, jd_text)
        else:
            return self._semantic_similarity_tfidf(resume_text, jd_text)

    def _semantic_similarity_transformers(
        self, resume_text: str, jd_text: str
    ) -> float:
        """Compute cosine similarity using sentence-transformers."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Truncate to avoid memory issues with very long texts
        max_chars = 5000
        resume_trunc = resume_text[:max_chars]
        jd_trunc = jd_text[:max_chars]

        embeddings = self.model.encode([resume_trunc, jd_trunc])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def _semantic_similarity_tfidf(
        self, resume_text: str, jd_text: str
    ) -> float:
        """Fallback: TF-IDF cosine similarity."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────────────────
    # ATS Keyword Analysis
    # ──────────────────────────────────────────────────────────────────────

    def ats_keyword_analysis(
        self, resume_text: str, jd_text: str
    ) -> tuple[list, list]:
        """
        Extract important ATS keywords from the JD and check which ones
        appear in the resume.

        Strategy:
        - Extract meaningful multi-word phrases from JD
        - Check for presence in resume
        - Prioritize nouns and technical terms

        Returns:
            Tuple of (matched_keywords, missing_keywords)
        """
        jd_keywords = self._extract_important_keywords(jd_text)
        resume_lower = resume_text.lower()

        matched = []
        missing = []

        for kw in jd_keywords:
            if kw.lower() in resume_lower:
                matched.append(kw)
            else:
                missing.append(kw)

        # Sort by length descending (longer phrases are more specific)
        matched = sorted(matched, key=len, reverse=True)
        missing = sorted(missing, key=len, reverse=True)

        return matched[:20], missing[:20]

    def _extract_important_keywords(self, text: str) -> list:
        """
        Extract candidate ATS keywords from text.
        Looks for capitalized terms, technical words, and multi-word phrases.
        """
        keywords = set()
        text_lower = text.lower()

        # 1. All known skills found in the JD
        for skill in SKILLS_DB:
            pattern = re.escape(skill)
            if re.search(pattern, text_lower):
                keywords.add(skill.title())

        # 2. Capitalized multi-word phrases (likely proper technical terms)
        cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        for phrase in cap_phrases:
            if 3 < len(phrase) < 50:
                keywords.add(phrase)

        # 3. All-caps acronyms (e.g., API, REST, SQL, CI/CD)
        acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
        for a in acronyms:
            if a not in {"THE", "AND", "FOR", "WITH", "ARE", "YOU", "CAN", "NOT", "ALL", "HAS"}:
                keywords.add(a)

        return sorted(keywords)[:30]

    # ──────────────────────────────────────────────────────────────────────
    # Experience Analysis
    # ──────────────────────────────────────────────────────────────────────

    def analyze_experience(self, resume_text: str, jd_text: str) -> dict:
        """
        Extract years of experience from both JD (requirement) and resume (actual).

        Returns:
            Dict with summary and any years found.
        """
        jd_years = self._extract_years_of_experience(jd_text)
        resume_years = self._extract_years_of_experience(resume_text)

        # Estimate resume experience from date spans
        estimated_years = self._estimate_experience_from_dates(resume_text)

        # Determine required years from JD
        required_years = max(jd_years) if jd_years else None

        # Determine actual years (use highest found or estimated)
        all_resume_years = resume_years + ([estimated_years] if estimated_years else [])
        actual_years = max(all_resume_years) if all_resume_years else None

        # Generate summary
        if required_years and actual_years:
            if actual_years >= required_years:
                summary = f"✅ You have ~{actual_years} years; requirement is {required_years}+ years."
            else:
                gap = required_years - actual_years
                summary = f"⚠️ You have ~{actual_years} years but {required_years}+ required ({gap} yr gap)."
        elif required_years:
            summary = f"📋 Job requires {required_years}+ years of experience."
        elif actual_years:
            summary = f"📄 Resume shows ~{actual_years} years of experience."
        else:
            summary = "Experience data could not be reliably extracted."

        return {
            "summary": summary,
            "required_years": required_years,
            "actual_years": actual_years,
            "years_found": [f"{y}yr" for y in sorted(set(jd_years + resume_years))],
        }

    def _extract_years_of_experience(self, text: str) -> list:
        """Extract numeric year values from experience mentions."""
        years = []
        text_lower = text.lower()
        for pattern in EXPERIENCE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    years.extend([int(m) for m in match if m.isdigit()])
                elif str(match).isdigit():
                    years.append(int(match))
        return years

    def _estimate_experience_from_dates(self, text: str) -> Optional[int]:
        """
        Estimate total work experience from year ranges in resume.
        E.g., '2018 – 2022' suggests 4 years.
        """
        year_ranges = re.findall(r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current)", text.lower())
        import datetime
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

        return total if total > 0 else None

    # ──────────────────────────────────────────────────────────────────────
    # Education Analysis
    # ──────────────────────────────────────────────────────────────────────

    def analyze_education(self, resume_text: str, jd_text: str) -> dict:
        """
        Detect degree requirements in JD and check resume for matching degrees.

        Returns:
            Dict with summary and detected degree levels.
        """
        resume_degrees = self._extract_degrees(resume_text)
        jd_degrees = self._extract_degrees(jd_text)

        degree_hierarchy = {
            "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
            "master": 4, "masters": 4, "m.s.": 4, "msc": 4, "m.tech": 4, "mba": 4,
            "bachelor": 3, "bachelors": 3, "b.s.": 3, "b.tech": 3, "bsc": 3, "b.e.": 3, "b.com": 3,
            "associate": 2, "diploma": 1,
        }

        # Highest degree in resume
        resume_max = max(
            (degree_hierarchy.get(d, 0) for d in resume_degrees), default=0
        )
        # Minimum degree required by JD
        jd_min = min(
            (degree_hierarchy.get(d, 0) for d in jd_degrees), default=0
        ) if jd_degrees else 0

        if resume_degrees and jd_degrees:
            if resume_max >= jd_min:
                summary = f"✅ Education meets requirements."
            else:
                summary = f"⚠️ Degree level may not meet JD requirement."
        elif resume_degrees:
            summary = f"📄 Degrees found in resume."
        elif jd_degrees:
            summary = f"📋 JD requires: {', '.join(jd_degrees[:3])}."
        else:
            summary = "No specific education requirement detected."

        return {
            "summary": summary,
            "resume_degrees": resume_degrees,
            "jd_degrees": jd_degrees,
            "degrees_found": [d.title() for d in (resume_degrees + jd_degrees)[:5]],
        }

    def _extract_degrees(self, text: str) -> list:
        """Extract degree mentions from text."""
        text_lower = text.lower()
        found = []
        for deg in DEGREE_KEYWORDS:
            if re.search(r"\b" + re.escape(deg) + r"\b", text_lower):
                found.append(deg)
        return found