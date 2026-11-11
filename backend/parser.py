"""
backend/parser.py
─────────────────
Handles PDF resume text extraction and text preprocessing.
Features robust multi-engine extraction (pdfplumber with automatic PyPDF2 fallback)
and unified NLP text normalization.
"""

import io
import re
import logging
from typing import Any, Dict, Optional
from utils.text_cleaner import clean_for_nlp

logger = logging.getLogger(__name__)


class ResumeParser:
    """
    Extracts text from PDF resumes and normalizes text for downstream analysis.
    Supports primary (pdfplumber) and secondary (PyPDF2) extraction engines.
    """

    def __init__(self):
        self.has_pdfplumber = self._check_module("pdfplumber")
        self.has_pypdf2 = self._check_module("PyPDF2")

        if not self.has_pdfplumber and not self.has_pypdf2:
            logger.warning("No PDF parser library installed (pdfplumber or PyPDF2).")

    @staticmethod
    def _check_module(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    def extract_text_from_pdf(self, pdf_source: Any) -> str:
        """
        Extract text from a PDF source (filepath, file-like object, or Streamlit UploadedFile).
        Uses pdfplumber by default with an automatic fallback to PyPDF2.
        """
        # Convert source to in-memory bytes stream
        if hasattr(pdf_source, "read"):
            content = pdf_source.read()
            # If Streamlit UploadedFile has a seek, rewind it as well
            if hasattr(pdf_source, "seek"):
                try:
                    pdf_source.seek(0)
                except Exception:
                    pass
            stream = io.BytesIO(content)
        elif isinstance(pdf_source, str):
            with open(pdf_source, "rb") as f:
                stream = io.BytesIO(f.read())
        elif isinstance(pdf_source, bytes):
            stream = io.BytesIO(pdf_source)
        else:
            raise ValueError("Unsupported PDF source type.")

        extracted_text = ""
        errors = []

        # Attempt 1: pdfplumber (best layout preservation)
        if self.has_pdfplumber:
            try:
                stream.seek(0)
                extracted_text = self._extract_with_pdfplumber(stream)
            except Exception as exc:
                logger.warning(f"pdfplumber extraction failed: {exc}. Trying fallback...")
                errors.append(f"pdfplumber: {exc}")

        # Attempt 2: PyPDF2 fallback if needed
        if not extracted_text.strip() and self.has_pypdf2:
            try:
                stream.seek(0)
                extracted_text = self._extract_with_pypdf2(stream)
            except Exception as exc:
                logger.warning(f"PyPDF2 fallback extraction failed: {exc}")
                errors.append(f"PyPDF2: {exc}")

        if not extracted_text.strip():
            error_details = "; ".join(errors) if errors else "No text found (scanned or image-based PDF)"
            raise ValueError(
                f"Could not extract readable text from PDF. Ensure the file contains selectable text. Details: {error_details}"
            )

        return extracted_text

    def _extract_with_pdfplumber(self, stream: io.BytesIO) -> str:
        import pdfplumber

        pages_text = []
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text)
        return "\n".join(pages_text)

    def _extract_with_pypdf2(self, stream: io.BytesIO) -> str:
        import PyPDF2

        pages_text = []
        reader = PyPDF2.PdfReader(stream)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text)
        return "\n".join(pages_text)

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw resume or JD text via unified cleaner."""
        return clean_for_nlp(text)

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Parses text into common resume sections (summary, experience, education, skills, projects).
        """
        sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "skills": "",
            "projects": "",
            "certifications": "",
            "other": "",
        }

        section_patterns = {
            "summary": r"^(?:professional\s+)?(?:summary|profile|about\s+me|career\s+objective)\b",
            "experience": r"^(?:work\s+)?(?:experience|employment\s+history|professional\s+background)\b",
            "education": r"^(?:education|academic\s+background|degrees?|qualifications?)\b",
            "skills": r"^(?:technical\s+)?(?:skills|core\s+competencies|technologies|expertise)\b",
            "projects": r"^(?:projects|portfolio|personal\s+projects|key\s+projects)\b",
            "certifications": r"^(?:certifications?|licenses?|credentials?)\b",
        }

        current_section = "other"
        lines = text.split("\n")

        for line in lines:
            line_str = line.strip()
            line_lower = line_str.lower()

            if not line_str:
                continue

            # Detect if line is a section heading (< 50 chars, no sentence ending)
            matched = False
            if len(line_str) < 50 and not line_str.endswith("."):
                for sec_key, pattern in section_patterns.items():
                    if re.search(pattern, line_lower):
                        current_section = sec_key
                        matched = True
                        break

            if not matched:
                sections[current_section] += line_str + "\n"

        return {k: v.strip() for k, v in sections.items()}
