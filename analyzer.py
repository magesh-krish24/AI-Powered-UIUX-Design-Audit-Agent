"""
analyzer.py - Gemini Vision client for the AI-powered UI/UX Design Audit Agent.

This module owns the single responsibility of taking a PIL Image, marshalling
it into a Gemini API request, and returning a validated Python dictionary that
conforms to the JSON_OUTPUT_SCHEMA defined in prompts.py.

All prompt engineering lives in prompts.py. All configuration lives in
config.py. This module is intentionally free of UI, framework, and I/O code.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai
from PIL import Image

from config import GEMINI_API_KEY, DEFAULT_MODEL_NAME, MIN_CONFIDENCE_SCORE
from prompts import SYSTEM_PROMPT, build_analysis_prompt

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gemini safety settings — keep all thresholds at their defaults so the SDK
# does not silently suppress responses for design-related content.
_SAFETY_SETTINGS: list[dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

_GENERATION_CONFIG: dict[str, Any] = {
    # Temperature 0 maximises determinism; design audits are analytical, not creative.
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    # Instruct the model to emit JSON so the SDK can surface finish_reason correctly.
    "response_mime_type": "application/json",
}

# Bytes → kilobytes conversion factor.
_BYTES_PER_KB: float = 1024.0

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AuditResult:
    """Typed wrapper around a completed design audit.

    Attributes:
        report:          The full audit dictionary conforming to JSON_OUTPUT_SCHEMA.
        filename:        Original filename of the analysed screenshot.
        model_name:      Gemini model that produced the report.
        latency_seconds: Wall-clock time (seconds) for the Gemini round-trip.
        raw_response:    The raw JSON string returned by the model, preserved for
                         debugging and logging purposes.
    """

    report:          dict[str, Any]
    filename:        str
    model_name:      str
    latency_seconds: float
    raw_response:    str


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AnalyzerError(Exception):
    """Base exception for all GeminiAnalyzer failures."""


class ImageValidationError(AnalyzerError):
    """Raised when the supplied image fails pre-flight validation."""


class APIError(AnalyzerError):
    """Raised when the Gemini API returns an error or an empty response."""


class ResponseParseError(AnalyzerError):
    """Raised when the model response cannot be parsed into valid JSON."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class GeminiAnalyzer:
    """Sends PIL images to Gemini Vision and returns structured audit reports.

    The class is intentionally stateless beyond the initialised SDK client so
    that a single instance can safely analyse multiple images sequentially
    (or concurrently, provided callers manage thread safety externally).

    Typical usage::

        analyzer = GeminiAnalyzer()
        image = Image.open("checkout.png")
        result = analyzer.analyze_image(image, filename="checkout.png")
        print(result.report["overall_score"])
    """

    def __init__(self) -> None:
        """Initialise the Gemini SDK client and model.

        Reads ``GEMINI_API_KEY`` and ``DEFAULT_MODEL_NAME`` from ``config.py``.

        Raises:
            AnalyzerError: If the Gemini SDK cannot be configured (e.g. the API
                key is rejected during client initialisation).
        """
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self._model: genai.GenerativeModel = genai.GenerativeModel(
                model_name=DEFAULT_MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
                generation_config=_GENERATION_CONFIG,
                safety_settings=_SAFETY_SETTINGS,
            )
        except Exception as exc:
            raise AnalyzerError(
                f"Failed to initialise Gemini SDK. "
                f"Verify that GEMINI_API_KEY is valid and the network is reachable. "
                f"Underlying error: {exc}"
            ) from exc

        logger.info(
            "GeminiAnalyzer initialised. model=%s min_confidence=%s",
            DEFAULT_MODEL_NAME,
            MIN_CONFIDENCE_SCORE,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image: Image.Image,
        filename: str,
        additional_context: str | None = None,
    ) -> AuditResult:
        """Analyse a PIL Image and return a structured design audit report.

        The method orchestrates the full pipeline:

        1. Pre-flight validation of the image object and filename.
        2. Calculation of the in-memory file size (used for prompt context).
        3. Construction of the user-turn prompt via ``build_analysis_prompt()``.
        4. Transmission of the image and prompt to Gemini Vision.
        5. Extraction and JSON-parsing of the model response.
        6. Post-processing: confidence filtering and metadata consistency check.
        7. Return of a fully populated :class:`AuditResult`.

        Args:
            image:              A ``PIL.Image.Image`` object. Any mode is
                                accepted; the image is converted to RGB
                                internally before being sent to the API.
            filename:           Original filename of the screenshot, used for
                                logging and embedded in the prompt for context.
                                Must be a non-empty string.
            additional_context: Optional free-text context from the submitting
                                user (e.g. target audience, area of concern).
                                Pass ``None`` to omit.

        Returns:
            An :class:`AuditResult` containing the parsed report dictionary,
            model metadata, and timing information.

        Raises:
            ImageValidationError: If ``image`` is not a PIL Image or
                                  ``filename`` is empty.
            APIError:             If the Gemini API returns an empty, blocked,
                                  or otherwise unusable response.
            ResponseParseError:   If the model response is not valid JSON or
                                  is missing required top-level keys.
            AnalyzerError:        For unexpected errors during the API call.
        """
        # ── Step 1: validate inputs ────────────────────────────────────────
        self._validate_image(image, filename)

        # ── Step 2: compute image size ─────────────────────────────────────
        file_size_kb = self._compute_size_kb(image)
        logger.debug("Image validated. filename=%s size_kb=%.1f", filename, file_size_kb)

        # ── Step 3: build prompt ───────────────────────────────────────────
        user_prompt = build_analysis_prompt(
            filename=filename,
            file_size_kb=file_size_kb,
            additional_context=additional_context,
        )

        # ── Step 4: call Gemini Vision ─────────────────────────────────────
        rgb_image = self._to_rgb(image)
        raw_response, latency = self._call_gemini(rgb_image, user_prompt, filename)

        # ── Step 5: parse JSON ─────────────────────────────────────────────
        report = self._parse_response(raw_response, filename)

        # ── Step 6: post-process ───────────────────────────────────────────
        report = self._filter_low_confidence(report)
        report = self._reconcile_metadata(report)

        # ── Step 7: return result ──────────────────────────────────────────
        logger.info(
            "Audit complete. filename=%s overall_score=%s findings=%s latency=%.2fs",
            filename,
            report.get("overall_score"),
            report.get("audit_metadata", {}).get("total_findings"),
            latency,
        )
        return AuditResult(
            report=report,
            filename=filename,
            model_name=DEFAULT_MODEL_NAME,
            latency_seconds=latency,
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: Any, filename: str) -> None:
        """Raise :class:`ImageValidationError` if inputs fail pre-flight checks.

        Args:
            image:    The value passed as the image argument.
            filename: The value passed as the filename argument.

        Raises:
            ImageValidationError: On any validation failure.
        """
        if not isinstance(image, Image.Image):
            raise ImageValidationError(
                f"Expected a PIL.Image.Image object; got {type(image).__name__}. "
                "Open the file with PIL.Image.open() before passing it to the analyzer."
            )
        if not filename or not filename.strip():
            raise ImageValidationError("'filename' must be a non-empty string.")

        width, height = image.size
        if width < 1 or height < 1:
            raise ImageValidationError(
                f"Image has zero-dimension size ({width}×{height}). "
                "The file may be corrupt or empty."
            )

        logger.debug("Image pre-flight passed. mode=%s size=%dx%d", image.mode, width, height)

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB mode if necessary.

        Gemini Vision handles JPEG-encoded images best in RGB. Modes like
        RGBA, P (palette), or L (greyscale) are converted transparently.

        Args:
            image: Source PIL image.

        Returns:
            The original image if already RGB, otherwise a converted copy.
        """
        if image.mode == "RGB":
            return image
        logger.debug("Converting image mode %s → RGB.", image.mode)
        return image.convert("RGB")

    @staticmethod
    def _compute_size_kb(image: Image.Image) -> float:
        """Estimate the in-memory PNG size of the image in kilobytes.

        The PNG size is used as a proxy for the original file size when the
        caller has not supplied a file path. It is embedded in the prompt for
        context and logged for observability.

        Args:
            image: PIL image (any mode).

        Returns:
            Estimated file size in kilobytes, rounded to one decimal place.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        size_kb = buffer.tell() / _BYTES_PER_KB
        return round(size_kb, 1)

    def _call_gemini(
        self,
        image: Image.Image,
        user_prompt: str,
        filename: str,
    ) -> tuple[str, float]:
        """Send the image and prompt to Gemini and return the raw text response.

        Args:
            image:       RGB PIL image.
            user_prompt: Fully constructed user-turn prompt string.
            filename:    Used for contextual log messages only.

        Returns:
            A ``(raw_text, latency_seconds)`` tuple.

        Raises:
            APIError:     If the response is blocked, empty, or the finish
                          reason indicates a non-successful completion.
            AnalyzerError: For unexpected SDK or network errors.
        """
        logger.info("Sending request to Gemini. filename=%s model=%s", filename, DEFAULT_MODEL_NAME)
        start = time.perf_counter()

        try:
            response = self._model.generate_content(
                contents=[image, user_prompt],
                stream=False,
            )
        except Exception as exc:
            raise AnalyzerError(
                f"Gemini API call failed for '{filename}'. "
                f"Check network connectivity and quota limits. "
                f"Underlying error: {exc}"
            ) from exc

        latency = time.perf_counter() - start
        logger.debug("Gemini responded in %.2fs.", latency)

        # ── Guard: blocked prompt ──────────────────────────────────────────
        if response.prompt_feedback and hasattr(response.prompt_feedback, "block_reason"):
            block_reason = response.prompt_feedback.block_reason
            if block_reason:
                raise APIError(
                    f"Gemini blocked the request for '{filename}'. "
                    f"block_reason={block_reason}. "
                    "Review the image content or adjust safety settings."
                )

        # ── Guard: no candidates ───────────────────────────────────────────
        if not response.candidates:
            raise APIError(
                f"Gemini returned no candidates for '{filename}'. "
                "The model may have encountered an internal error."
            )

        candidate = response.candidates[0]

        # ── Guard: non-STOP finish reason ──────────────────────────────────
        finish_reason = getattr(candidate, "finish_reason", None)
        # finish_reason == 1 corresponds to FinishReason.STOP in the SDK enum.
        if finish_reason is not None and finish_reason != 1:
            logger.warning(
                "Unexpected finish_reason=%s for '%s'. Attempting to parse anyway.",
                finish_reason,
                filename,
            )

        # ── Extract text ───────────────────────────────────────────────────
        try:
            raw_text: str = response.text
        except ValueError as exc:
            raise APIError(
                f"Could not extract text from Gemini response for '{filename}'. "
                f"The response may have been filtered. Details: {exc}"
            ) from exc

        if not raw_text or not raw_text.strip():
            raise APIError(
                f"Gemini returned an empty response for '{filename}'. "
                "Retry or check the model configuration."
            )

        return raw_text.strip(), latency

    @staticmethod
    def _parse_response(raw_response: str, filename: str) -> dict[str, Any]:
        """Parse the raw Gemini response string into a Python dictionary.

        The method strips common LLM artefacts (markdown fences) before
        parsing to make the pipeline resilient to minor model formatting
        deviations.

        Args:
            raw_response: Raw text returned by :meth:`_call_gemini`.
            filename:     Used for contextual error messages.

        Returns:
            Parsed report dictionary.

        Raises:
            ResponseParseError: If the text is not valid JSON or is missing
                                 the required top-level keys.
        """
        # Strip markdown code fences that some model versions include.
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove opening fence (```json or ```) and closing fence (```)
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            report: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse failure for '%s'. offset=%d snippet=%r",
                filename,
                exc.pos,
                cleaned[max(0, exc.pos - 40): exc.pos + 40],
            )
            raise ResponseParseError(
                f"Model response for '{filename}' is not valid JSON. "
                f"Parse error at position {exc.pos}: {exc.msg}. "
                "Check raw_response on the AuditResult for the raw model output."
            ) from exc

        # ── Structural validation ──────────────────────────────────────────
        required_keys = {"screenshot_summary", "overall_score", "findings", "audit_metadata"}
        missing = required_keys - report.keys()
        if missing:
            raise ResponseParseError(
                f"Model response for '{filename}' is missing required top-level "
                f"keys: {sorted(missing)}. The model may have deviated from the schema."
            )

        if not isinstance(report.get("findings"), list):
            raise ResponseParseError(
                f"'findings' in the model response for '{filename}' is not a list. "
                f"Got: {type(report.get('findings')).__name__}."
            )

        logger.debug(
            "Parsed response successfully. filename=%s findings=%d",
            filename,
            len(report["findings"]),
        )
        return report

    @staticmethod
    def _filter_low_confidence(report: dict[str, Any]) -> dict[str, Any]:
        """Remove findings whose confidence_score falls below MIN_CONFIDENCE_SCORE.

        Although the schema specifies ``minimum: 60`` for confidence_score,
        this post-processing step enforces the project-level threshold
        (``MIN_CONFIDENCE_SCORE`` from config.py) as a second safety net.

        Args:
            report: Parsed report dictionary.

        Returns:
            The same report dictionary with low-confidence findings removed.
            Mutates ``report["findings"]`` in place and returns the report for
            convenient chaining.
        """
        # Convert 0–1 float threshold (from config) to 0–100 integer scale.
        threshold = int(MIN_CONFIDENCE_SCORE * 100)
        original_count = len(report["findings"])

        report["findings"] = [
            f for f in report["findings"]
            if isinstance(f.get("confidence_score"), (int, float))
            and f["confidence_score"] >= threshold
        ]

        removed = original_count - len(report["findings"])
        if removed:
            logger.info(
                "Filtered %d low-confidence finding(s) below threshold=%d.",
                removed,
                threshold,
            )
        return report

    @staticmethod
    def _reconcile_metadata(report: dict[str, Any]) -> dict[str, Any]:
        """Recompute audit_metadata counts from the actual findings list.

        The model is instructed to self-report counts, but this step overwrites
        those counts with values derived from the authoritative findings array.
        This eliminates any discrepancy the model may have introduced.

        Args:
            report: Parsed (and confidence-filtered) report dictionary.

        Returns:
            The same report dictionary with audit_metadata counts corrected.
        """
        findings: list[dict[str, Any]] = report.get("findings", [])

        severity_counts: dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0,
        }
        principle_counts: dict[str, int] = {
            "visual_hierarchy": 0,
            "contrast_wcag_aa": 0,
            "spacing": 0,
            "alignment": 0,
            "consistency": 0,
        }

        for finding in findings:
            sev = finding.get("severity", "")
            if sev in severity_counts:
                severity_counts[sev] += 1

            prin = finding.get("principle", "")
            if prin in principle_counts:
                principle_counts[prin] += 1

        metadata: dict[str, Any] = report.setdefault("audit_metadata", {})
        metadata["total_findings"]        = len(findings)
        metadata["findings_by_severity"]  = severity_counts
        metadata["findings_by_principle"] = principle_counts
        # Preserve model_notes if the model supplied them; default to empty string.
        metadata.setdefault("model_notes", "")

        logger.debug("Metadata reconciled. total_findings=%d", len(findings))
        return report
