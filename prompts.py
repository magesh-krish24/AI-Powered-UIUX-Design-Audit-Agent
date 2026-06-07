"""
prompts.py - Prompt templates and schema definitions for the AI-powered
UI/UX Design Audit Agent.

Level 1: SYSTEM_PROMPT, JSON_OUTPUT_SCHEMA, build_analysis_prompt()
Level 2: COMPARISON_SYSTEM_PROMPT, COMPARISON_OUTPUT_SCHEMA, build_comparison_prompt()
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Final

# ---------------------------------------------------------------------------
# ── LEVEL 1: SINGLE SCREENSHOT AUDIT (unchanged) ───────────────────────────
# ---------------------------------------------------------------------------

JSON_OUTPUT_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DesignAuditReport",
    "type": "object",
    "required": ["screenshot_summary", "overall_score", "findings", "audit_metadata"],
    "additionalProperties": False,
    "properties": {
        "screenshot_summary": {"type": "string", "minLength": 20, "maxLength": 400},
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "principle_scores": {
            "type": "object",
            "properties": {
                "visual_hierarchy": {"type": "integer", "minimum": 0, "maximum": 100},
                "contrast_wcag_aa": {"type": "integer", "minimum": 0, "maximum": 100},
                "spacing":          {"type": "integer", "minimum": 0, "maximum": 100},
                "alignment":        {"type": "integer", "minimum": 0, "maximum": 100},
                "consistency":      {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"],
            "additionalProperties": False,
        },
        "findings": {
            "type": "array",
            "minItems": 0,
            "items": {
                "type": "object",
                "required": ["id", "principle", "location", "issue", "user_impact", "recommendation", "severity", "confidence_score"],
                "additionalProperties": False,
                "properties": {
                    "id":               {"type": "string", "pattern": "^F-\\d{3}$"},
                    "principle":        {"type": "string", "enum": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"]},
                    "location":         {"type": "string", "minLength": 10, "maxLength": 200},
                    "issue":            {"type": "string", "minLength": 20, "maxLength": 500},
                    "user_impact":      {"type": "string", "minLength": 20, "maxLength": 400},
                    "recommendation":   {"type": "string", "minLength": 20, "maxLength": 500},
                    "severity":         {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
                    "confidence_score": {"type": "integer", "minimum": 60, "maximum": 100},
                },
            },
        },
        "audit_metadata": {
            "type": "object",
            "required": ["total_findings", "findings_by_severity", "findings_by_principle", "model_notes"],
            "additionalProperties": False,
            "properties": {
                "total_findings":        {"type": "integer", "minimum": 0},
                "findings_by_severity":  {"type": "object", "properties": {"critical": {"type": "integer"}, "high": {"type": "integer"}, "medium": {"type": "integer"}, "low": {"type": "integer"}, "info": {"type": "integer"}}, "required": ["critical", "high", "medium", "low", "info"], "additionalProperties": False},
                "findings_by_principle": {"type": "object", "properties": {"visual_hierarchy": {"type": "integer"}, "contrast_wcag_aa": {"type": "integer"}, "spacing": {"type": "integer"}, "alignment": {"type": "integer"}, "consistency": {"type": "integer"}}, "required": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"], "additionalProperties": False},
                "model_notes":           {"type": "string", "maxLength": 600},
            },
        },
    },
}

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

    1. VISUAL HIERARCHY  — size, weight, colour relationships; focal-point clarity.
    2. CONTRAST (WCAG AA) — normal text ≥4.5:1; large text / UI components ≥3.0:1.
    3. SPACING           — padding, margin, line-height; touch target ≥44×44px.
    4. ALIGNMENT         — shared axes; ragged edges; baseline grids.
    5. CONSISTENCY       — button styles, type scale, icon set, border-radius, palette.

    ═══════════════════════════════════════════════════════════════════════
    STRICT OPERATING RULES
    ═══════════════════════════════════════════════════════════════════════
    ✦ Only report issues directly observable in the screenshot.
    ✦ Do not hallucinate issues about hover states, breakpoints, or dynamic content.
    ✦ When issues exist, identify a minimum of 3 distinct findings.
    ✦ Include only findings where confidence_score ≥ 60.
    ✦ Recommendations must be specific and actionable with concrete values.
    ✦ Return ONLY a valid JSON object — no markdown fences, no prose.
""")


def build_analysis_prompt(
    *,
    filename: str,
    file_size_kb: float,
    additional_context: str | None = None,
) -> str:
    """Build the user-turn prompt for a Level 1 single-screenshot audit.

    Args:
        filename:           Original filename of the screenshot.
        file_size_kb:       Size of the screenshot in kilobytes.
        additional_context: Optional free-text context from the user.

    Returns:
        Fully assembled prompt string ready to send to Gemini.

    Raises:
        ValueError: If filename is empty or file_size_kb is negative.
    """
    if not filename or not filename.strip():
        raise ValueError("'filename' must be a non-empty string.")
    if file_size_kb < 0:
        raise ValueError(f"'file_size_kb' must be non-negative; got {file_size_kb}.")

    schema_block = json.dumps(JSON_OUTPUT_SCHEMA, indent=2)

    context_block = ""
    if additional_context and additional_context.strip():
        context_block = dedent(f"""\

            ── DESIGNER-SUPPLIED CONTEXT ─────────────────────────────────────────
            {additional_context.strip()}
            ──────────────────────────────────────────────────────────────────────
        """)

    return dedent(f"""\
        Conduct a full design audit of the attached screenshot.

        ── FILE METADATA ────────────────────────────────────────────────────
        Filename : {filename.strip()}
        File size: {file_size_kb:.1f} KB
        ─────────────────────────────────────────────────────────────────────
        {context_block}
        ── REQUIRED OUTPUT SCHEMA ───────────────────────────────────────────
        Return ONLY a JSON object matching this schema exactly:

        {schema_block}
        ─────────────────────────────────────────────────────────────────────

        Analyse the screenshot now and return only the JSON audit report.
    """)


# ---------------------------------------------------------------------------
# ── LEVEL 2: DESIGN COMPARISON (new) ───────────────────────────────────────
# ---------------------------------------------------------------------------

# The JSON schema that Gemini must follow for the comparison response.
# Each list item (improvement, regression, etc.) is a simple object with
# a principle, description, and impact — keeping it easy to render in the UI.
COMPARISON_OUTPUT_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DesignComparisonReport",
    "type": "object",
    "required": [
        "before_score",
        "after_score",
        "score_change",
        "improvements",
        "regressions",
        "new_issues",
        "resolved_issues",
        "overall_verdict",
        "summary",
    ],
    "additionalProperties": False,
    "properties": {
        # Numeric scores for each design (0–100)
        "before_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "after_score":  {"type": "integer", "minimum": 0, "maximum": 100},
        # Positive = improved, negative = degraded, zero = no change
        "score_change": {"type": "integer", "minimum": -100, "maximum": 100},

        # Things that got better in the After screenshot
        "improvements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["principle", "description", "impact"],
                "additionalProperties": False,
                "properties": {
                    "principle":   {"type": "string", "enum": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"]},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                    "impact":      {"type": "string", "minLength": 10, "maxLength": 300},
                },
            },
        },

        # Things that got worse in the After screenshot
        "regressions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["principle", "description", "impact"],
                "additionalProperties": False,
                "properties": {
                    "principle":   {"type": "string", "enum": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"]},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                    "impact":      {"type": "string", "minLength": 10, "maxLength": 300},
                },
            },
        },

        # Issues visible in After that were NOT present in Before
        "new_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["principle", "description", "severity"],
                "additionalProperties": False,
                "properties": {
                    "principle":   {"type": "string", "enum": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"]},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                    "severity":    {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
                },
            },
        },

        # Issues visible in Before that are no longer visible in After
        "resolved_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["principle", "description"],
                "additionalProperties": False,
                "properties": {
                    "principle":   {"type": "string", "enum": ["visual_hierarchy", "contrast_wcag_aa", "spacing", "alignment", "consistency"]},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                },
            },
        },

        # One of three possible verdicts
        "overall_verdict": {
            "type": "string",
            "enum": ["Improved", "Degraded", "No Significant Change"],
        },

        # 2–4 sentence plain-English summary of the comparison
        "summary": {"type": "string", "minLength": 30, "maxLength": 600},
    },
}


# The system prompt for Level 2 comparison mode.
# It tells the model it is comparing two screenshots, not auditing one.
COMPARISON_SYSTEM_PROMPT: Final[str] = dedent("""\
    You are a senior UI/UX design auditor specialising in before-and-after
    design comparisons. You evaluate pairs of screenshots to determine whether
    a redesign has improved or degraded the interface.

    ═══════════════════════════════════════════════════════════════════════
    MISSION
    ═══════════════════════════════════════════════════════════════════════
    You will receive TWO screenshots:
      • IMAGE 1 = the BEFORE design (the original).
      • IMAGE 2 = the AFTER design (the redesign or updated version).

    Compare them across five design principles and return a structured
    comparison report. Base every finding on directly observable visual
    evidence only. Do not speculate or assume.

    ═══════════════════════════════════════════════════════════════════════
    FIVE PRINCIPLES TO COMPARE
    ═══════════════════════════════════════════════════════════════════════
    1. VISUAL HIERARCHY  — Does the After version guide the eye more clearly?
    2. CONTRAST (WCAG AA)— Did contrast ratios improve or worsen?
    3. SPACING           — Is spacing more consistent and intentional?
    4. ALIGNMENT         — Are elements better aligned?
    5. CONSISTENCY       — Is the visual language more unified?

    ═══════════════════════════════════════════════════════════════════════
    STRICT OPERATING RULES
    ═══════════════════════════════════════════════════════════════════════
    ✦ Only report changes that are directly visible between the two images.
    ✦ score_change must equal after_score minus before_score exactly.
    ✦ overall_verdict must match score_change:
        - Positive score_change  → "Improved"
        - Negative score_change  → "Degraded"
        - Zero score_change      → "No Significant Change"

    ═══════════════════════════════════════════════════════════════════════
    OUTPUT FORMAT — READ CAREFULLY
    ═══════════════════════════════════════════════════════════════════════
    YOUR RESPONSE MUST BE RAW JSON AND NOTHING ELSE.

    FORBIDDEN — your response must never contain any of these:
      ✗ Markdown code fences  (``` or ```json or ```JSON)
      ✗ Any text before the opening brace  {
      ✗ Any text after the closing brace   }
      ✗ Single-quoted property names  ('key' is wrong, "key" is correct)
      ✗ Trailing commas inside objects or arrays
      ✗ Comments of any kind  (// or /* */ are not valid JSON)
      ✗ Explanations, apologies, or notes outside the JSON object

    REQUIRED — your response must:
      ✓ Start with the character  {  as the very first character
      ✓ End with the character    }  as the very last character
      ✓ Use double quotes around every property name and string value
      ✓ Be parseable by Python's  json.loads()  with zero modification
""")


def build_comparison_prompt(
    *,
    before_filename: str,
    after_filename: str,
    before_size_kb: float,
    after_size_kb: float,
    additional_context: str | None = None,
) -> str:
    """Build the user-turn prompt for a Level 2 design comparison.

    The model receives this text prompt alongside two images. The prompt
    tells it which image is Before and which is After, and embeds the
    full JSON schema so the model knows exactly what to return.

    Args:
        before_filename:    Original filename of the Before screenshot.
        after_filename:     Original filename of the After screenshot.
        before_size_kb:     Size of the Before image in kilobytes.
        after_size_kb:      Size of the After image in kilobytes.
        additional_context: Optional free-text context from the user.

    Returns:
        Fully assembled comparison prompt string.

    Raises:
        ValueError: If either filename is empty or sizes are negative.
    """
    if not before_filename or not before_filename.strip():
        raise ValueError("'before_filename' must be a non-empty string.")
    if not after_filename or not after_filename.strip():
        raise ValueError("'after_filename' must be a non-empty string.")
    if before_size_kb < 0 or after_size_kb < 0:
        raise ValueError("File sizes must be non-negative.")

    schema_block = json.dumps(COMPARISON_OUTPUT_SCHEMA, indent=2)

    context_block = ""
    if additional_context and additional_context.strip():
        context_block = dedent(f"""\

            ── ADDITIONAL CONTEXT ────────────────────────────────────────────────
            {additional_context.strip()}
            ──────────────────────────────────────────────────────────────────────
        """)

    return dedent(f"""\
        Compare the two attached screenshots: IMAGE 1 is the BEFORE design,
        IMAGE 2 is the AFTER design.

        ── FILE METADATA ────────────────────────────────────────────────────
        Before : {before_filename.strip()} ({before_size_kb:.1f} KB)
        After  : {after_filename.strip()} ({after_size_kb:.1f} KB)
        ─────────────────────────────────────────────────────────────────────
        {context_block}
        ── REQUIRED OUTPUT SCHEMA ───────────────────────────────────────────
        Your response MUST be a single raw JSON object — nothing else.
        No markdown fences. No ```json. No text before {{ or after }}.
        Every property name must use double quotes.
        The response must be parseable by json.loads() with no modification.

        Match this schema exactly:

        {schema_block}
        ─────────────────────────────────────────────────────────────────────

        Respond now. Start your response with the character {{ and end with }}.
    """)

# ---------------------------------------------------------------------------
# ── LEVEL 3: PRODUCT UX AUDIT (new) ───────────────────────────────────────
# ---------------------------------------------------------------------------

PRODUCT_UX_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ProductUXAuditReport",
    "type": "object",
    "required": [
        "overall_score",
        "consistency_score",
        "screens_analyzed",
        "summary",
        "strengths",
        "issues",
        "recommendations",
        "final_verdict",
    ],
    "additionalProperties": False,
    "properties": {
        "overall_score":      {"type": "integer", "minimum": 0, "maximum": 100},
        "consistency_score":  {"type": "integer", "minimum": 0, "maximum": 100},
        "screens_analyzed":   {"type": "integer", "minimum": 1},
        "summary":            {"type": "string", "minLength": 30, "maxLength": 600},
        "strengths": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["area", "description"],
                "additionalProperties": False,
                "properties": {
                    "area":        {"type": "string", "minLength": 3, "maxLength": 80},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                },
            },
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["area", "description", "severity", "affected_screens"],
                "additionalProperties": False,
                "properties": {
                    "area":        {"type": "string", "minLength": 3, "maxLength": 80},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                    "severity":    {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "affected_screens": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "description", "priority"],
                "additionalProperties": False,
                "properties": {
                    "title":       {"type": "string", "minLength": 3, "maxLength": 100},
                    "description": {"type": "string", "minLength": 10, "maxLength": 400},
                    "priority":    {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
        "final_verdict": {"type": "string", "minLength": 10, "maxLength": 300},
    },
}

PRODUCT_UX_SYSTEM_PROMPT: Final[str] = dedent("""
    You are a senior product UX auditor who specialises in evaluating the
    design consistency and usability of multi-screen digital products.

    ═══════════════════════════════════════════════════════════════════════
    MISSION
    ═══════════════════════════════════════════════════════════════════════
    You will receive multiple screenshots from a single product. Evaluate them
    together as a cohesive product experience. Base every finding on directly
    observable visual evidence only.

    ═══════════════════════════════════════════════════════════════════════
    STRICT OPERATING RULES
    ═══════════════════════════════════════════════════════════════════════
    ✦ Evaluate ALL provided screenshots together, not each in isolation.
    ✦ Focus on cross-screen inconsistencies and product UX issues.
    ✦ Only report issues that are observable in the screenshots.
    ✦ screens_analyzed must equal the number of images provided.
    ✦ Identify at least 2 strengths and at least 2 issues when they exist.

    ═══════════════════════════════════════════════════════════════════════
    OUTPUT FORMAT — READ CAREFULLY
    ═══════════════════════════════════════════════════════════════════════
    YOUR RESPONSE MUST BE RAW JSON AND NOTHING ELSE.

    FORBIDDEN:
      ✗ Markdown fences (``` or ```json)
      ✗ Any text before the opening brace  {
      ✗ Any text after the closing brace   }
      ✗ Single-quoted property names
      ✗ Trailing commas or comments

    REQUIRED:
      ✓ Start with { as the very first character
      ✓ End with } as the very last character
      ✓ Use double quotes around every property name and string value
      ✓ Be parseable by json.loads() with zero modification
""")


def build_product_ux_prompt(*, filenames: list[str], total_size_kb: float) -> str:
    """Build the user-turn prompt for a Level 3 product UX audit."""
    if not filenames:
        raise ValueError("'filenames' must contain at least one filename.")

    schema_block = json.dumps(PRODUCT_UX_SCHEMA, indent=2)
    screen_list = "\n".join(
        f"  Screen {i + 1}: {name}" for i, name in enumerate(filenames)
    )

    return dedent(f"""
        Perform a product UX audit across all attached screenshots.

        ── SCREENS PROVIDED ─────────────────────────────────────────────────
        {screen_list}
        Total size: {total_size_kb:.1f} KB
        ─────────────────────────────────────────────────────────────────────
        Return ONLY a JSON object that matches this schema exactly:

        {schema_block}
        ─────────────────────────────────────────────────────────────────────
        Analyse the product UX now and return only the JSON report.
    """)
