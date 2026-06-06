"""
prompts.py - Prompt templates and schema definitions for the AI-powered
UI/UX Design Audit Agent.

This module centralises every instruction that is sent to the Gemini vision
model so that prompt engineering lives in one auditable place, separate from
business logic.

Design philosophy
-----------------
* Ground truth only — the model is explicitly forbidden from inferring issues
  that are not directly observable in the provided screenshot.
* Schema-first — a canonical JSON schema is defined here and embedded verbatim
  inside the prompt so the model has an unambiguous contract to honour.
* Severity & confidence as first-class citizens — the model is required to
  quantify both for every finding so downstream code can filter and prioritise
  without re-invoking the model.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Final

# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------

JSON_OUTPUT_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DesignAuditReport",
    "description": (
        "Structured output returned by the Design Audit Agent for a single "
        "screenshot. All findings are grounded in visible evidence only."
    ),
    "type": "object",
    "required": ["screenshot_summary", "overall_score", "findings", "audit_metadata"],
    "additionalProperties": False,
    "properties": {
        "screenshot_summary": {
            "type": "string",
            "description": (
                "1–3 sentence neutral description of what the screenshot shows "
                "(interface type, apparent purpose, primary UI elements). "
                "No evaluative language."
            ),
            "minLength": 20,
            "maxLength": 400,
        },
        "overall_score": {
            "type": "integer",
            "description": (
                "Holistic design quality score from 0 (completely broken) to "
                "100 (exemplary). Derived from the weighted average of "
                "per-principle scores, penalised by critical/high findings."
            ),
            "minimum": 0,
            "maximum": 100,
        },
        "principle_scores": {
            "type": "object",
            "description": "Per-principle score breakdown (0–100 each).",
            "properties": {
                "visual_hierarchy": {"type": "integer", "minimum": 0, "maximum": 100},
                "contrast_wcag_aa": {"type": "integer", "minimum": 0, "maximum": 100},
                "spacing":          {"type": "integer", "minimum": 0, "maximum": 100},
                "alignment":        {"type": "integer", "minimum": 0, "maximum": 100},
                "consistency":      {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": [
                "visual_hierarchy",
                "contrast_wcag_aa",
                "spacing",
                "alignment",
                "consistency",
            ],
            "additionalProperties": False,
        },
        "findings": {
            "type": "array",
            "description": (
                "Ordered list of design issues, sorted by severity "
                "(critical → high → medium → low → info) then by confidence "
                "descending. Must contain at least 3 items when issues exist."
            ),
            "minItems": 0,
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "principle",
                    "location",
                    "issue",
                    "user_impact",
                    "recommendation",
                    "severity",
                    "confidence_score",
                ],
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "type": "string",
                        "description": (
                            "Sequential finding identifier in the format "
                            "'F-001', 'F-002', etc."
                        ),
                        "pattern": "^F-\\d{3}$",
                    },
                    "principle": {
                        "type": "string",
                        "description": "The design principle this finding relates to.",
                        "enum": [
                            "visual_hierarchy",
                            "contrast_wcag_aa",
                            "spacing",
                            "alignment",
                            "consistency",
                        ],
                    },
                    "location": {
                        "type": "string",
                        "description": (
                            "Precise, human-readable location of the issue in the "
                            "screenshot. Reference visible landmarks, labels, or "
                            "screen regions (e.g. 'Primary CTA button in the hero "
                            "section', 'Navigation bar – right cluster of icons'). "
                            "Do not use pixel coordinates."
                        ),
                        "minLength": 10,
                        "maxLength": 200,
                    },
                    "issue": {
                        "type": "string",
                        "description": (
                            "Clear, factual description of the problem that is "
                            "directly observable in the screenshot. Use present "
                            "tense. Do not speculate."
                        ),
                        "minLength": 20,
                        "maxLength": 500,
                    },
                    "user_impact": {
                        "type": "string",
                        "description": (
                            "Explanation of how this issue affects real users — "
                            "readability, discoverability, accessibility, task "
                            "completion, trust, etc."
                        ),
                        "minLength": 20,
                        "maxLength": 400,
                    },
                    "recommendation": {
                        "type": "string",
                        "description": (
                            "Specific, actionable fix a designer or developer can "
                            "implement. Reference concrete values where possible "
                            "(e.g. 'Increase contrast ratio to at least 4.5:1 by "
                            "darkening the text colour to #595959 or darker')."
                        ),
                        "minLength": 20,
                        "maxLength": 500,
                    },
                    "severity": {
                        "type": "string",
                        "description": dedent("""\
                            Severity tier:
                            - critical : Blocks task completion or causes WCAG failure
                            - high     : Significantly degrades usability or accessibility
                            - medium   : Noticeable friction; degrades experience
                            - low      : Minor polish issue with limited real-world impact
                            - info     : Observation or best-practice suggestion only\
                        """),
                        "enum": ["critical", "high", "medium", "low", "info"],
                    },
                    "confidence_score": {
                        "type": "integer",
                        "description": (
                            "Your confidence that this issue truly exists and is "
                            "directly visible in the screenshot (0 = no confidence, "
                            "100 = absolute certainty). Only include findings with "
                            "a score ≥ 60."
                        ),
                        "minimum": 60,
                        "maximum": 100,
                    },
                },
            },
        },
        "audit_metadata": {
            "type": "object",
            "description": "Bookkeeping fields produced by the model.",
            "required": [
                "total_findings",
                "findings_by_severity",
                "findings_by_principle",
                "model_notes",
            ],
            "additionalProperties": False,
            "properties": {
                "total_findings": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Total number of findings in this report.",
                },
                "findings_by_severity": {
                    "type": "object",
                    "description": "Count of findings at each severity level.",
                    "properties": {
                        "critical": {"type": "integer", "minimum": 0},
                        "high":     {"type": "integer", "minimum": 0},
                        "medium":   {"type": "integer", "minimum": 0},
                        "low":      {"type": "integer", "minimum": 0},
                        "info":     {"type": "integer", "minimum": 0},
                    },
                    "required": ["critical", "high", "medium", "low", "info"],
                    "additionalProperties": False,
                },
                "findings_by_principle": {
                    "type": "object",
                    "description": "Count of findings under each design principle.",
                    "properties": {
                        "visual_hierarchy": {"type": "integer", "minimum": 0},
                        "contrast_wcag_aa": {"type": "integer", "minimum": 0},
                        "spacing":          {"type": "integer", "minimum": 0},
                        "alignment":        {"type": "integer", "minimum": 0},
                        "consistency":      {"type": "integer", "minimum": 0},
                    },
                    "required": [
                        "visual_hierarchy",
                        "contrast_wcag_aa",
                        "spacing",
                        "alignment",
                        "consistency",
                    ],
                    "additionalProperties": False,
                },
                "model_notes": {
                    "type": "string",
                    "description": (
                        "Optional free-text notes from the model — e.g. image "
                        "quality caveats, partial visibility of UI elements, or "
                        "anything that may affect audit completeness. "
                        "Empty string if no caveats."
                    ),
                    "maxLength": 600,
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = dedent("""\
    You are a senior UI/UX design auditor with deep expertise in WCAG 2.1
    accessibility guidelines, visual design principles, and human-computer
    interaction. You conduct rigorous, evidence-based audits of digital
    interfaces.

    ═══════════════════════════════════════════════════════════════════════
    MISSION
    ═══════════════════════════════════════════════════════════════════════
    Analyse the provided screenshot and produce a structured design audit
    report. Your sole source of evidence is what is directly visible in the
    screenshot. You must not infer, assume, or speculate about anything that
    cannot be seen.

    ═══════════════════════════════════════════════════════════════════════
    FIVE DESIGN PRINCIPLES YOU EVALUATE
    ═══════════════════════════════════════════════════════════════════════

    1. VISUAL HIERARCHY
       Assess whether the layout guides the user's eye in an intentional
       sequence. Look for: size relationships between headings, body text,
       and captions; use of weight and colour to differentiate importance;
       focal-point clarity; competing elements at the same visual weight
       that dilute hierarchy.

    2. CONTRAST (WCAG AA)
       Measure observable contrast between text/UI elements and their
       backgrounds against WCAG 2.1 AA thresholds:
         • Normal text (< 18 pt / 14 pt bold) → minimum ratio 4.5 : 1
         • Large text  (≥ 18 pt / 14 pt bold) → minimum ratio 3.0 : 1
         • UI components & graphical objects   → minimum ratio 3.0 : 1
       Report estimated ratios where you can reasonably estimate them from
       visible colours. Flag all violations.

    3. SPACING
       Evaluate padding, margin, and line-height consistency. Look for:
       crowded or cramped elements; inconsistent gaps between sibling
       components; insufficient touch/click target sizes (< 44 × 44 px
       guideline); missing breathing room around section boundaries.

    4. ALIGNMENT
       Inspect whether elements share clear alignment axes. Look for:
       elements that are almost — but not exactly — aligned; mixed
       left/centre/right alignment without apparent intent; ragged edges
       in grids or lists; broken baseline grids in typographic content.

    5. CONSISTENCY
       Check for visual language uniformity. Look for: inconsistent
       button styles within the same context; mixed type scales (too many
       distinct font sizes); inconsistent icon styles (filled vs. outlined
       mixed arbitrarily); differing border-radius values on similar
       components; colour usage that deviates from an apparent palette.

    ═══════════════════════════════════════════════════════════════════════
    STRICT OPERATING RULES
    ═══════════════════════════════════════════════════════════════════════

    EVIDENCE
    ✦ Only report issues that are unambiguously observable in the screenshot.
    ✦ Do not hallucinate issues that could theoretically exist but are not
      visible (e.g. do not assume hover states, dynamic content, or
      responsive breakpoints that are not shown).
    ✦ If image quality limits your analysis, note it in model_notes.

    FINDINGS
    ✦ When design issues exist, identify a minimum of 3 findings.
    ✦ Each finding must address a genuinely distinct problem — do not split
      a single issue artificially to meet the minimum.
    ✦ Include only findings where confidence_score ≥ 60.
    ✦ Every finding must contain all seven required fields: principle,
      location, issue, user_impact, recommendation, severity,
      confidence_score.
    ✦ Recommendations must be specific and actionable, not generic advice.
    ✦ Explain user_impact in terms of real user experience consequences
      (readability, accessibility, task completion, trust).

    SCORING
    ✦ overall_score must reflect the true state of the UI — do not inflate
      scores to appear favourable.
    ✦ A single critical finding should reduce overall_score by at least 15
      points from an otherwise clean baseline.

    OUTPUT
    ✦ Return ONLY a valid JSON object that conforms exactly to the schema
      provided in the user message. No markdown fences, no prose, no
      preamble, no trailing commentary — raw JSON only.
    ✦ Ensure all counts in audit_metadata are consistent with the findings
      array (total_findings, findings_by_severity, findings_by_principle).
""")

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    *,
    filename: str,
    file_size_kb: float,
    additional_context: str | None = None,
) -> str:
    """Construct the user-turn prompt for a single design audit request.

    This function combines:

    * The canonical JSON output schema (serialised inline so the model has an
      unambiguous structural contract).
    * File metadata to give the model grounding context.
    * An optional designer-supplied context string (e.g. target audience,
      known constraints, or a specific area of concern).
    * A closing instruction that reinforces the JSON-only output requirement.

    Args:
        filename: Original filename of the screenshot being audited.  Used
            purely for contextual grounding; no file I/O is performed here.
        file_size_kb: Size of the screenshot in kilobytes, included so the
            model can note if the image appears heavily compressed.
        additional_context: Free-text context supplied by the submitting user
            (e.g. "This is a mobile checkout flow targeting first-time buyers"
            or "Focus especially on the navigation area").  Pass ``None`` or
            an empty string to omit.

    Returns:
        A fully-formed prompt string ready to be sent as the ``user`` turn
        in the Gemini API request alongside the screenshot image part.

    Raises:
        ValueError: If ``filename`` is empty or ``file_size_kb`` is negative.

    Example::

        prompt = build_analysis_prompt(
            filename="checkout_v3.png",
            file_size_kb=184.5,
            additional_context="Mobile viewport, targeting accessibility-conscious users.",
        )
    """
    if not filename or not filename.strip():
        raise ValueError("'filename' must be a non-empty string.")
    if file_size_kb < 0:
        raise ValueError(f"'file_size_kb' must be non-negative; got {file_size_kb}.")

    import json  # local import — json is stdlib, deferred to avoid circular issues

    schema_block = json.dumps(JSON_OUTPUT_SCHEMA, indent=2)

    context_block = ""
    if additional_context and additional_context.strip():
        context_block = dedent(f"""\

            ── DESIGNER-SUPPLIED CONTEXT ────────────────────────────────────────
            {additional_context.strip()}
            ─────────────────────────────────────────────────────────────────────
        """)

    prompt = dedent(f"""\
        Conduct a full design audit of the attached screenshot.

        ── FILE METADATA ────────────────────────────────────────────────────
        Filename : {filename.strip()}
        File size: {file_size_kb:.1f} KB
        ─────────────────────────────────────────────────────────────────────
        {context_block}
        ── REQUIRED OUTPUT SCHEMA ───────────────────────────────────────────
        Your response MUST be a single JSON object that conforms exactly to
        the schema below. Do not emit markdown fences, prose, or any text
        outside the JSON object.

        {schema_block}
        ─────────────────────────────────────────────────────────────────────

        Analyse the screenshot now and return only the JSON audit report.
    """)

    return prompt
