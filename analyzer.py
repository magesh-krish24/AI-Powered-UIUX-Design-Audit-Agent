"""
analyzer.py - Gemini Vision client for the AI-powered UI/UX Design Audit Agent.

Level 1: GeminiAnalyzer.analyze_image()   — single screenshot audit.
Level 2: GeminiAnalyzer.compare_designs() — before/after comparison (new).

All new code is additive. Nothing from Level 1 has been changed.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
from PIL import Image

from config import GEMINI_API_KEY, DEFAULT_MODEL_NAME, MIN_CONFIDENCE_SCORE
from prompts import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    # ── Level 2 additions ──
    COMPARISON_SYSTEM_PROMPT,
    build_comparison_prompt,
    # ── Level 3 additions ──
    PRODUCT_UX_SYSTEM_PROMPT,
    build_product_ux_prompt,
)

logger = logging.getLogger(__name__)

# ── Gemini generation settings (shared by both modes) ─────────────────────────
_SAFETY_SETTINGS: list[dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

_GENERATION_CONFIG: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

_BYTES_PER_KB: float = 1024.0


# ── Return type for Level 1 ────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class AuditResult:
    """Typed wrapper around a completed single-screenshot audit.

    Attributes:
        report:          Parsed audit dictionary (conforms to JSON_OUTPUT_SCHEMA).
        filename:        Original filename of the analysed screenshot.
        model_name:      Gemini model that produced the report.
        latency_seconds: Wall-clock time for the Gemini round-trip.
        raw_response:    Raw JSON string returned by the model.
    """
    report:          dict[str, Any]
    filename:        str
    model_name:      str
    latency_seconds: float
    raw_response:    str


# ── Return type for Level 2 (new) ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Typed wrapper around a completed before/after design comparison.

    Attributes:
        report:           Parsed comparison dictionary (conforms to COMPARISON_OUTPUT_SCHEMA).
        before_filename:  Original filename of the Before screenshot.
        after_filename:   Original filename of the After screenshot.
        model_name:       Gemini model that produced the report.
        latency_seconds:  Wall-clock time for the Gemini round-trip.
        raw_response:     Raw JSON string returned by the model.
    """
    report:          dict[str, Any]
    before_filename: str
    after_filename:  str
    model_name:      str
    latency_seconds: float
    raw_response:    str


# ── Return type for Level 3 (new) ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ProductAuditResult:
    """Typed wrapper around a completed product UX audit.

    Attributes:
        report:          Parsed product audit dictionary.
        filenames:       List of original filenames that were analysed.
        model_name:      Gemini model that produced the report.
        latency_seconds: Wall-clock time for the Gemini round-trip.
        raw_response:    Raw JSON string returned by the model.
    """
    report:          dict[str, Any]
    filenames:       list[str]
    model_name:      str
    latency_seconds: float
    raw_response:    str


# ── Custom exceptions (unchanged from Level 1) ─────────────────────────────────
class AnalyzerError(Exception):
    """Base exception for all GeminiAnalyzer failures."""

class ImageValidationError(AnalyzerError):
    """Raised when the supplied image fails pre-flight validation."""

class APIError(AnalyzerError):
    """Raised when the Gemini API returns an error or an empty response."""

class ResponseParseError(AnalyzerError):
    """Raised when the model response cannot be parsed into valid JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# GeminiAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class GeminiAnalyzer:
    """Sends PIL images to Gemini Vision and returns structured audit reports.

    Supports two modes:
      • Level 1 — analyze_image()   : single screenshot audit.
      • Level 2 — compare_designs() : before/after design comparison.

    A single instance can handle both modes. The comparison method creates
    a separate Gemini model instance internally so that it can use its own
    system prompt without affecting the Level 1 model.
    """

    def __init__(self) -> None:
        """Initialise the Level 1 Gemini model client.

        The Level 2 model is initialised lazily inside compare_designs()
        the first time it is needed, so __init__ stays fast and simple.

        Raises:
            AnalyzerError: If the Gemini SDK cannot be configured.
        """
        try:
            genai.configure(api_key=GEMINI_API_KEY)

            # ── Level 1 model (single screenshot audit) ────────────────────
            self._model = genai.GenerativeModel(
                model_name=DEFAULT_MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
                generation_config=_GENERATION_CONFIG,
                safety_settings=_SAFETY_SETTINGS,
            )

            # ── Level 2 model (comparison) — created lazily ─────────────────
            # We keep it as None here and create it only when compare_designs()
            # is first called. This avoids creating two SDK clients on startup.
            self._comparison_model = None

            # ── Level 3 model (product UX audit) — created lazily ───────────
            self._product_model = None

        except Exception as exc:
            raise AnalyzerError(
                f"Failed to initialise Gemini SDK. "
                f"Verify that GEMINI_API_KEY is valid. Error: {exc}"
            ) from exc

        logger.info("GeminiAnalyzer initialised. model=%s", DEFAULT_MODEL_NAME)

    # ── Private helper: get (or create) the Level 2 model ─────────────────────
    def _get_comparison_model(self):
        """Return the Level 2 comparison model, creating it on first call.

        Using a separate model instance for comparison means we can give it
        its own system prompt (COMPARISON_SYSTEM_PROMPT) without changing
        the Level 1 model's instructions.
        """
        if self._comparison_model is None:
            self._comparison_model = genai.GenerativeModel(
                model_name=DEFAULT_MODEL_NAME,
                system_instruction=COMPARISON_SYSTEM_PROMPT,
                generation_config=_GENERATION_CONFIG,
                safety_settings=_SAFETY_SETTINGS,
            )
            logger.info("Comparison model initialised.")
        return self._comparison_model

    # ── Private helper: get (or create) the Level 3 model ─────────────────────
    def _get_product_model(self):
        """Return the Level 3 product-audit model, creating it on first call."""
        if self._product_model is None:
            self._product_model = genai.GenerativeModel(
                model_name=DEFAULT_MODEL_NAME,
                system_instruction=PRODUCT_UX_SYSTEM_PROMPT,
                generation_config=_GENERATION_CONFIG,
                safety_settings=_SAFETY_SETTINGS,
            )
            logger.info("Product UX audit model initialised.")
        return self._product_model

    # =========================================================================
    # LEVEL 1 — Single Screenshot Audit (unchanged)
    # =========================================================================

    def analyze_image(
        self,
        image: Image.Image,
        filename: str,
        additional_context: str | None = None,
    ) -> AuditResult:
        """Analyse a PIL Image and return a structured design audit report.

        Args:
            image:              PIL Image object of the screenshot.
            filename:           Original filename of the screenshot.
            additional_context: Optional context text from the user.

        Returns:
            AuditResult containing the parsed report and metadata.

        Raises:
            ImageValidationError: If the image or filename is invalid.
            APIError:             If the Gemini API response is unusable.
            ResponseParseError:   If the response is not valid JSON.
        """
        self._validate_image(image, filename)
        file_size_kb = self._compute_size_kb(image)

        user_prompt = build_analysis_prompt(
            filename=filename,
            file_size_kb=file_size_kb,
            additional_context=additional_context,
        )

        rgb_image = self._to_rgb(image)
        raw_response, latency = self._call_gemini(
            model=self._model,
            contents=[rgb_image, user_prompt],
            filename=filename,
        )

        report = self._parse_response(raw_response, filename)
        report = self._filter_low_confidence(report)
        report = self._reconcile_metadata(report)

        logger.info(
            "Audit complete. filename=%s score=%s findings=%s latency=%.2fs",
            filename, report.get("overall_score"),
            report.get("audit_metadata", {}).get("total_findings"), latency,
        )
        return AuditResult(
            report=report,
            filename=filename,
            model_name=DEFAULT_MODEL_NAME,
            latency_seconds=latency,
            raw_response=raw_response,
        )

    # =========================================================================
    # LEVEL 2 — Design Comparison (new method)
    # =========================================================================

    def compare_designs(
        self,
        before_image: Image.Image,
        after_image: Image.Image,
        before_filename: str,
        after_filename: str,
        additional_context: str | None = None,
    ) -> ComparisonResult:
        """Compare a Before and After screenshot and return a comparison report.

        This method sends BOTH images to Gemini in a single request. The model
        sees Image 1 (Before) and Image 2 (After) side-by-side in its context
        and evaluates what changed across the five design principles.

        Args:
            before_image:       PIL Image of the original/before design.
            after_image:        PIL Image of the redesigned/after design.
            before_filename:    Original filename of the Before screenshot.
            after_filename:     Original filename of the After screenshot.
            additional_context: Optional context text from the user.

        Returns:
            ComparisonResult containing the parsed comparison report.

        Raises:
            ImageValidationError: If either image or filename is invalid.
            APIError:             If the Gemini API response is unusable.
            ResponseParseError:   If the response is not valid JSON.
        """
        # Validate both images before making any API call.
        self._validate_image(before_image, before_filename)
        self._validate_image(after_image, after_filename)

        before_size_kb = self._compute_size_kb(before_image)
        after_size_kb  = self._compute_size_kb(after_image)

        # Build the prompt that explains which image is Before and which is After.
        user_prompt = build_comparison_prompt(
            before_filename=before_filename,
            after_filename=after_filename,
            before_size_kb=before_size_kb,
            after_size_kb=after_size_kb,
            additional_context=additional_context,
        )

        # Convert both images to RGB (Gemini handles RGB best).
        before_rgb = self._to_rgb(before_image)
        after_rgb  = self._to_rgb(after_image)

        # Send: [before_image, after_image, text_prompt] in one request.
        # The model's system prompt tells it Image 1 = Before, Image 2 = After.
        raw_response, latency = self._call_gemini(
            model=self._get_comparison_model(),
            contents=[before_rgb, after_rgb, user_prompt],
            filename=f"{before_filename} vs {after_filename}",
        )

        # Parse the JSON response — same helper as Level 1.
        report = self._parse_comparison_response(raw_response, before_filename, after_filename)

        logger.info(
            "Comparison complete. before=%s after=%s verdict=%s latency=%.2fs",
            before_filename, after_filename,
            report.get("overall_verdict"), latency,
        )
        return ComparisonResult(
            report=report,
            before_filename=before_filename,
            after_filename=after_filename,
            model_name=DEFAULT_MODEL_NAME,
            latency_seconds=latency,
            raw_response=raw_response,
        )

    # =========================================================================
    # LEVEL 3 — Product UX Audit (new method)
    # =========================================================================

    def analyze_product_flow(
        self,
        images: list[Image.Image],
        filenames: list[str],
    ) -> ProductAuditResult:
        """Audit multiple product screenshots together for UX consistency."""
        if not images:
            raise ImageValidationError("At least one image is required for a product audit.")
        if len(images) != len(filenames):
            raise ImageValidationError(
                f"images ({len(images)}) and filenames ({len(filenames)}) must be the same length."
            )
        for img, fname in zip(images, filenames):
            self._validate_image(img, fname)

        rgb_images = [self._to_rgb(img) for img in images]
        total_size_kb = sum(self._compute_size_kb(img) for img in rgb_images)

        user_prompt = build_product_ux_prompt(
            filenames=filenames,
            total_size_kb=total_size_kb,
        )

        contents = rgb_images + [user_prompt]
        raw_response, latency = self._call_gemini(
            model=self._get_product_model(),
            contents=contents,
            filename=f"{len(images)} screens",
        )

        report = self._parse_product_response(raw_response, filenames)

        logger.info(
            "Product audit complete. screens=%d score=%s latency=%.2fs",
            len(images), report.get("overall_score"), latency,
        )
        return ProductAuditResult(
            report=report,
            filenames=filenames,
            model_name=DEFAULT_MODEL_NAME,
            latency_seconds=latency,
            raw_response=raw_response,
        )

    @staticmethod
    def _parse_product_response(
        raw_response: str,
        filenames: list[str],
    ) -> dict[str, Any]:
        """Parse the Level 3 product audit JSON response."""
        json_str = GeminiAnalyzer._extract_json(raw_response)

        try:
            report: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            snippet = json_str[max(0, exc.pos - 60): exc.pos + 60]
            logger.error("JSON parse error in product response. pos=%d snippet=%r", exc.pos, snippet)
            raise ResponseParseError(
                f"Product audit response for {len(filenames)} screens is not valid JSON. "
                f"Error at position {exc.pos}: {exc.msg}. Area: {snippet!r}"
            ) from exc

        required = {
            "overall_score", "consistency_score", "screens_analyzed",
            "summary", "strengths", "issues", "recommendations", "final_verdict",
        }
        missing = required - report.keys()
        if missing:
            raise ResponseParseError(
                f"Product audit response is missing keys: {sorted(missing)}."
            )

        return report

    # =========================================================================
    # Private helpers (shared by both levels)
    # =========================================================================

    @staticmethod
    def _validate_image(image: Any, filename: str) -> None:
        """Check that the image is a valid PIL Image and filename is non-empty."""
        if not isinstance(image, Image.Image):
            raise ImageValidationError(
                f"Expected PIL.Image.Image; got {type(image).__name__}. "
                "Use PIL.Image.open() before passing to the analyzer."
            )
        if not filename or not filename.strip():
            raise ImageValidationError("'filename' must be a non-empty string.")
        width, height = image.size
        if width < 1 or height < 1:
            raise ImageValidationError(
                f"Image has zero-dimension size ({width}×{height}). File may be corrupt."
            )

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB mode. Gemini Vision works best with RGB."""
        if image.mode == "RGB":
            return image
        return image.convert("RGB")

    @staticmethod
    def _compute_size_kb(image: Image.Image) -> float:
        """Estimate the in-memory PNG size of the image in kilobytes."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return round(buffer.tell() / _BYTES_PER_KB, 1)

    @staticmethod
    def _call_gemini(model, contents: list, filename: str) -> tuple[str, float]:
        """Send a request to Gemini and return (raw_text, latency_seconds).

        This helper is reused by both analyze_image() and compare_designs().
        The caller passes the right model instance and the right contents list.

        Args:
            model:    The GenerativeModel instance to use.
            contents: List of content parts (images and/or text strings).
            filename: Used only for log and error messages.

        Returns:
            Tuple of (raw_text_response, latency_in_seconds).

        Raises:
            APIError:     If the response is blocked or empty.
            AnalyzerError: For unexpected SDK or network errors.
        """
        logger.info("Sending request to Gemini. context=%s", filename)
        start = time.perf_counter()

        try:
            response = model.generate_content(contents=contents, stream=False)
        except Exception as exc:
            raise AnalyzerError(
                f"Gemini API call failed for '{filename}'. Error: {exc}"
            ) from exc

        latency = time.perf_counter() - start

        # Guard: blocked prompt
        if response.prompt_feedback and getattr(response.prompt_feedback, "block_reason", None):
            raise APIError(
                f"Gemini blocked the request for '{filename}'. "
                f"block_reason={response.prompt_feedback.block_reason}"
            )

        # Guard: no candidates returned
        if not response.candidates:
            raise APIError(f"Gemini returned no candidates for '{filename}'.")

        # Guard: empty text
        try:
            raw_text: str = response.text
        except ValueError as exc:
            raise APIError(
                f"Could not extract text from Gemini response for '{filename}'. {exc}"
            ) from exc

        if not raw_text or not raw_text.strip():
            raise APIError(f"Gemini returned an empty response for '{filename}'.")

        return raw_text.strip(), latency

    @staticmethod
    def _parse_response(raw_response: str, filename: str) -> dict[str, Any]:
        """Parse Level 1 JSON response. Strips markdown fences if present."""
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        try:
            report: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ResponseParseError(
                f"Model response for '{filename}' is not valid JSON. "
                f"Error at position {exc.pos}: {exc.msg}."
            ) from exc

        required = {"screenshot_summary", "overall_score", "findings", "audit_metadata"}
        missing = required - report.keys()
        if missing:
            raise ResponseParseError(
                f"Response for '{filename}' is missing keys: {sorted(missing)}."
            )

        return report

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the first complete JSON object out of a raw string.

        This is the core safety net for the comparison parser. It handles
        every messy thing Gemini might prepend or append to the JSON:

        • Markdown fences  (```json ... ``` or ``` ... ```)
        • Introductory prose  ("Here is the comparison report: {...")
        • Trailing notes     ("} \n\nLet me know if you need anything.")
        • Mixed indentation or leading whitespace on the fence line

        The strategy is simple:
          1. Strip markdown fences line-by-line.
          2. Find the first '{' and the last '}' in what remains.
          3. Return only the text between those two characters (inclusive).

        Args:
            text: Raw string returned by the Gemini API.

        Returns:
            The extracted JSON substring, ready for json.loads().

        Raises:
            ResponseParseError: If no '{' or '}' can be found at all.
        """
        # Step 1 — remove every line that is only a markdown fence.
        # A fence line looks like: optional-whitespace ``` optional-language-tag
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just ``` or ```json or ```JSON etc.
            if stripped.startswith("```"):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)

        # Step 2 — find the outer braces of the JSON object.
        start = cleaned.find("{")
        end   = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ResponseParseError(
                "Could not find a JSON object in the model response. "
                "The response may be pure prose or was completely malformed. "
                f"First 200 chars received: {text[:200]!r}"
            )

        # Step 3 — slice out exactly the JSON object.
        return cleaned[start : end + 1]

    @staticmethod
    def _parse_comparison_response(
        raw_response: str,
        before_filename: str,
        after_filename: str,
    ) -> dict[str, Any]:
        """Parse the Level 2 comparison JSON response into a Python dictionary.

        Uses _extract_json() first to safely isolate the JSON object from any
        surrounding prose or markdown that Gemini might have added, then runs
        json.loads(), then validates the required top-level keys.

        Args:
            raw_response:    Raw text returned by _call_gemini().
            before_filename: Used in error messages only.
            after_filename:  Used in error messages only.

        Returns:
            Parsed comparison report dictionary.

        Raises:
            ResponseParseError: If the text cannot be parsed or is missing keys.
        """
        label = f"'{before_filename}' vs '{after_filename}'"

        # Pull out just the JSON object, discarding any surrounding text.
        try:
            json_str = GeminiAnalyzer._extract_json(raw_response)
        except ResponseParseError:
            raise  # re-raise with the message already set by _extract_json

        # Parse the isolated JSON string.
        try:
            report: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            # Log a snippet around the problem position to help debugging.
            snippet_start = max(0, exc.pos - 60)
            snippet_end   = min(len(json_str), exc.pos + 60)
            snippet = json_str[snippet_start:snippet_end]
            logger.error(
                "JSON parse error in comparison response. pos=%d snippet=%r",
                exc.pos, snippet,
            )
            raise ResponseParseError(
                f"Comparison response for {label} is not valid JSON. "
                f"Parse error at position {exc.pos}: {exc.msg}. "
                f"Problem area: {snippet!r}"
            ) from exc

        # Validate that all required top-level keys are present.
        required = {
            "before_score", "after_score", "score_change",
            "improvements", "regressions", "new_issues",
            "resolved_issues", "overall_verdict", "summary",
        }
        missing = required - report.keys()
        if missing:
            raise ResponseParseError(
                f"Comparison response for {label} is missing required keys: "
                f"{sorted(missing)}. Keys present: {sorted(report.keys())}."
            )

        # Fix any arithmetic error in score_change silently.
        expected_change = report["after_score"] - report["before_score"]
        if report["score_change"] != expected_change:
            logger.warning(
                "score_change mismatch: model said %d, expected %d. Correcting.",
                report["score_change"], expected_change,
            )
            report["score_change"] = expected_change

        return report

    @staticmethod
    def _filter_low_confidence(report: dict[str, Any]) -> dict[str, Any]:
        """Remove Level 1 findings below MIN_CONFIDENCE_SCORE threshold."""
        threshold = int(MIN_CONFIDENCE_SCORE * 100)
        original_count = len(report["findings"])
        report["findings"] = [
            f for f in report["findings"]
            if isinstance(f.get("confidence_score"), (int, float))
            and f["confidence_score"] >= threshold
        ]
        removed = original_count - len(report["findings"])
        if removed:
            logger.info("Filtered %d low-confidence finding(s).", removed)
        return report

    @staticmethod
    def _reconcile_metadata(report: dict[str, Any]) -> dict[str, Any]:
        """Recompute Level 1 audit_metadata counts from the actual findings list."""
        findings = report.get("findings", [])
        severity_counts  = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        principle_counts = {"visual_hierarchy": 0, "contrast_wcag_aa": 0, "spacing": 0, "alignment": 0, "consistency": 0}

        for f in findings:
            sev  = f.get("severity", "")
            prin = f.get("principle", "")
            if sev  in severity_counts:  severity_counts[sev]   += 1
            if prin in principle_counts: principle_counts[prin] += 1

        metadata = report.setdefault("audit_metadata", {})
        metadata["total_findings"]        = len(findings)
        metadata["findings_by_severity"]  = severity_counts
        metadata["findings_by_principle"] = principle_counts
        metadata.setdefault("model_notes", "")
        return report