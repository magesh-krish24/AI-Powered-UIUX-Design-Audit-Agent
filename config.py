"""
config.py - Configuration module for the AI-powered UI/UX Design Audit Agent.

Loads environment variables, validates required secrets, and exposes
typed constants used throughout the application.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, FrozenSet

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

# Resolve the .env file relative to this file's location so the module works
# regardless of the working directory from which the application is launched.
_ENV_PATH: Final[Path] = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ---------------------------------------------------------------------------
# Secrets & API keys
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Return the value of an environment variable or raise a descriptive error.

    Args:
        name: The name of the environment variable to look up.

    Returns:
        The non-empty string value of the variable.

    Raises:
        ValueError: If the variable is missing or empty.
    """
    value: str | None = os.getenv(name)
    if not value or not value.strip():
        raise ValueError(
            f"[Design Audit Agent] Required environment variable '{name}' is "
            f"missing or empty.\n"
            f"  • Make sure a '.env' file exists at: {_ENV_PATH}\n"
            f"  • Add the line:  {name}=<your-api-key>\n"
            f"  • Never commit real API keys to version control."
        )
    return value.strip()


GEMINI_API_KEY: Final[str] = _require_env("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "gemini-2.5-flash"
"""The Gemini model used for design analysis unless overridden at runtime."""


# ---------------------------------------------------------------------------
# Image / file constraints
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_FORMATS: Final[FrozenSet[str]] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
})
"""File extensions accepted by the audit pipeline (lowercase, dot-prefixed)."""

MAX_FILE_SIZE_MB: Final[float] = 10.0
"""Maximum allowed upload size in megabytes before the file is rejected."""

MAX_FILE_SIZE_BYTES: Final[int] = int(MAX_FILE_SIZE_MB * 1024 * 1024)
"""``MAX_FILE_SIZE_MB`` expressed in bytes for direct comparison with ``os.path.getsize``."""


# ---------------------------------------------------------------------------
# Analysis thresholds
# ---------------------------------------------------------------------------

MIN_CONFIDENCE_SCORE: Final[float] = 0.60
"""Minimum confidence (0.0 – 1.0) for a finding to be included in the report.

Findings whose confidence falls below this threshold are silently discarded so
that low-signal noise does not clutter the output presented to designers.
"""


# ---------------------------------------------------------------------------
# Audit categories
# ---------------------------------------------------------------------------

AUDIT_CATEGORIES: Final[tuple[str, ...]] = (
    "visual_hierarchy",
    "contrast_wcag_aa",
    "spacing",
    "alignment",
    "consistency",
)
"""Ordered list of design-quality dimensions evaluated during each audit."""


# ---------------------------------------------------------------------------
# WCAG contrast thresholds (AA level)
# ---------------------------------------------------------------------------

WCAG_AA_NORMAL_TEXT_RATIO: Final[float] = 4.5
"""Minimum contrast ratio required for normal text under WCAG 2.1 AA."""

WCAG_AA_LARGE_TEXT_RATIO: Final[float] = 3.0
"""Minimum contrast ratio required for large text (≥ 18 pt / 14 pt bold) under WCAG 2.1 AA."""

WCAG_AA_UI_COMPONENT_RATIO: Final[float] = 3.0
"""Minimum contrast ratio required for UI components and graphical objects under WCAG 2.1 AA."""


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def is_supported_format(filename: str) -> bool:
    """Return *True* if *filename* has an extension accepted by the agent.

    Args:
        filename: The original file name (with or without a leading path).

    Returns:
        ``True`` when the lowercase file extension is in
        :data:`SUPPORTED_IMAGE_FORMATS`, ``False`` otherwise.

    Example::

        >>> is_supported_format("dashboard.PNG")
        True
        >>> is_supported_format("report.pdf")
        False
    """
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_FORMATS


def is_within_size_limit(file_path: str | Path) -> bool:
    """Return *True* if the file at *file_path* does not exceed :data:`MAX_FILE_SIZE_BYTES`.

    Args:
        file_path: Absolute or relative path to the file on disk.

    Returns:
        ``True`` when the file size is within the allowed limit.

    Raises:
        FileNotFoundError: If *file_path* does not point to an existing file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path.stat().st_size <= MAX_FILE_SIZE_BYTES
