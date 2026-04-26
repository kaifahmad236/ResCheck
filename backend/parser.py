"""
backend/parser.py
─────────────────
Handles PDF resume text extraction and text preprocessing.
Supports both pdfplumber (preferred) and PyPDF2 as fallback.
"""

import re
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ResumeParser:
    """
    Responsible for:
    - Extracting raw text from PDF resumes
    - Cleaning and normalizing text for NLP processing
    """

    def __init__(self):
        self._check_pdf_libraries()

    def _check_pdf_libraries(self):
        """Check which PDF library is available."""
        self.use_pdfplumber = False
        try:
            import pdfplumber
            self.use_pdfplumber = True
            logger.info("Using pdfplumber for PDF extraction.")
        except ImportError:
            try:
                import PyPDF2
                logger.info("pdfplumber not found. Falling back to PyPDF2.")
            except ImportError:
                raise ImportError(
                    "No PDF library found. Install with: pip install pdfplumber"
                )

    # ──────────────────────────────────────────────────────────────────────
    # PDF Extraction
    # ──────────────────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_source) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_source: File path (str), file-like object, or Streamlit UploadedFile

        Returns:
            Raw extracted text as a string.
        """
        # Convert Streamlit UploadedFile or file-like to bytes
        if hasattr(pdf_source, "read"):
            pdf_bytes = pdf_source.read()
            pdf_source = io.BytesIO(pdf_bytes)
        elif isinstance(pdf_source, str):
            with open(pdf_source, "rb") as f:
                pdf_bytes = f.read()
            pdf_source = io.BytesIO(pdf_bytes)

        if self.use_pdfplumber:
            return self._extract_with_pdfplumber(pdf_source)
        else:
            return self._extract_with_pypdf2(pdf_source)

    def _extract_with_pdfplumber(self, pdf_source) -> str:
        """Extract text using pdfplumber (better formatting preservation)."""
        import pdfplumber

        all_text = []
        try:
            with pdfplumber.open(pdf_source) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise ValueError(f"Failed to read PDF: {e}")

        if not all_text:
            raise ValueError("No text could be extracted from the PDF. The file may be image-based or corrupted.")

        return "\n".join(all_text)

    def _extract_with_pypdf2(self, pdf_source) -> str:
        """Extract text using PyPDF2 (fallback)."""
        import PyPDF2

        all_text = []
        try:
            reader = PyPDF2.PdfReader(pdf_source)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise ValueError(f"Failed to read PDF: {e}")

        if not all_text:
            raise ValueError("No text could be extracted from the PDF.")

        return "\n".join(all_text)

    # ──────────────────────────────────────────────────────────────────────
    # Text Cleaning
    # ──────────────────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text extracted from a resume or job description.

        Steps:
        1. Decode unicode artifacts
        2. Remove special characters
        3. Normalize whitespace
        4. Remove very short lines (noise)

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Fix common unicode issues
        text = text.encode("ascii", errors="ignore").decode("ascii")

        # Replace non-breaking spaces and odd whitespace chars
        text = re.sub(r"[\xa0\u200b\t\r]", " ", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove phone numbers
        text = re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", "", text)

        # Remove excessive special characters (keep hyphens, forward slashes, dots, commas)
        text = re.sub(r"[^\w\s,./\-+#()&]", " ", text)

        # Normalize multiple spaces to single
        text = re.sub(r"[ ]{2,}", " ", text)

        # Normalize multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip each line and remove very short lines (likely noise)
        lines = text.split("\n")
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 2]

        return "\n".join(cleaned_lines).strip()

    def extract_sections(self, text: str) -> dict:
        """
        Attempt to split resume text into common sections.

        Args:
            text: Cleaned resume text

        Returns:
            Dictionary with section names as keys and content as values.
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

        # Patterns to detect common resume section headers
        section_patterns = {
            "summary": r"(summary|objective|profile|about me|career objective)",
            "experience": r"(experience|work history|employment|professional background|work experience)",
            "education": r"(education|academic|qualification|schooling|degree)",
            "skills": r"(skills|technical skills|core competencies|technologies|expertise|proficiencies)",
            "projects": r"(projects|portfolio|work samples|achievements)",
            "certifications": r"(certifications|certificates|licenses|accreditations)",
        }

        lines = text.split("\n")
        current_section = "other"

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line is a section header
            matched_section = None
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line.strip()) < 60:
                    matched_section = section
                    break

            if matched_section:
                current_section = matched_section
            else:
                sections[current_section] += line + "\n"

        # Strip extra whitespace from all sections
        return {k: v.strip() for k, v in sections.items()}