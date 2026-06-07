"""
app.py - Streamlit UI for the AI-Powered UI/UX Design Audit Agent.

Level 1: Single screenshot audit (unchanged).
Level 2: Before/After design comparison (new).

A mode selector at the top lets the user switch between the two modes.
Everything in Level 1 is identical to the original — no existing code was removed.
"""

import streamlit as st
from PIL import Image

from analyzer import GeminiAnalyzer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="UI/UX Design Audit Agent", layout="centered")
st.title("🎨 AI-Powered UI/UX Design Audit Agent")

# ── Mode selector ──────────────────────────────────────────────────────────────
# This is the only new element added to the top of the page.
# The user picks between Level 1 (single audit) and Level 2 (comparison).
mode = st.radio(
    label="Select Mode",
    options=["🔍 Level 1: Design Audit", "⚖️ Level 2: Design Comparison"],
    horizontal=True,
)

st.divider()

# =============================================================================
# LEVEL 1 — Single Screenshot Audit (original code, completely unchanged)
# =============================================================================
if mode == "🔍 Level 1: Design Audit":

    st.write("Upload a screenshot and the AI will audit it for design issues.")

    # Step 1: File uploader
    uploaded_file = st.file_uploader(
        label="Upload a screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="level1_upload",   # unique key so Streamlit doesn't mix up the two uploaders
    )

    # Step 2: Optional context
    additional_context = st.text_area(
        label="Additional Context (optional)",
        placeholder="e.g. This is a mobile checkout screen targeting new users.",
        height=80,
        key="level1_context",
    )

    # Step 3: Show the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Screenshot", use_container_width=True)

    # Step 4: Analyze button
    analyze_clicked = st.button("🔍 Analyze Design", type="primary", key="level1_btn")

    if analyze_clicked:
        if uploaded_file is None:
            st.error("Please upload a screenshot before clicking Analyze.")
        else:
            image = Image.open(uploaded_file)
            with st.spinner("Analyzing your design with Gemini Vision..."):
                try:
                    analyzer = GeminiAnalyzer()
                    result = analyzer.analyze_image(
                        image=image,
                        filename=uploaded_file.name,
                        additional_context=additional_context or None,
                    )
                    report = result.report
                except Exception as error:
                    st.error(f"Analysis failed: {error}")
                    st.stop()

            # Step 5: Show results
            st.success("Analysis complete!")

            st.subheader("📋 Screenshot Summary")
            st.write(report.get("screenshot_summary", "No summary available."))

            overall_score = report.get("overall_score", 0)
            st.subheader("📊 Overall Design Score")
            if overall_score >= 80:
                st.write(f"✅ {overall_score} / 100 — Good")
            elif overall_score >= 60:
                st.write(f"⚠️ {overall_score} / 100 — Needs Work")
            else:
                st.write(f"❌ {overall_score} / 100 — Poor")
            st.progress(overall_score / 100)

            findings = report.get("findings", [])
            st.subheader(f"🔎 Findings ({len(findings)} issue(s) found)")

            if len(findings) == 0:
                st.write("No issues found. Great design!")

            severity_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}

            for finding in findings:
                severity  = finding.get("severity", "unknown").upper()
                principle = finding.get("principle", "unknown").replace("_", " ").title()
                issue     = finding.get("issue", "No description.")
                location  = finding.get("location", "Unknown location")
                impact    = finding.get("user_impact", "No impact described.")
                rec       = finding.get("recommendation", "No recommendation.")
                confidence = finding.get("confidence_score", 0)
                icon = severity_icons.get(severity, "❓")

                with st.expander(f"{icon} [{severity}] {principle} — {location}"):
                    st.write(f"**Issue:** {issue}")
                    st.write(f"**User Impact:** {impact}")
                    st.write(f"**Recommendation:** {rec}")
                    st.write(f"**Confidence Score:** {confidence} / 100")

            st.caption(f"⏱️ Analysis completed in {result.latency_seconds:.2f} seconds.")


# =============================================================================
# LEVEL 2 — Design Comparison (all new code below)
# =============================================================================
else:

    st.write("Upload a **Before** and **After** screenshot to compare the two designs.")

    # Step 1: Two side-by-side file uploaders
    col_before, col_after = st.columns(2)

    with col_before:
        before_file = st.file_uploader(
            label="📸 Before Screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            key="before_upload",
        )

    with col_after:
        after_file = st.file_uploader(
            label="📸 After Screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            key="after_upload",
        )

    # Step 2: Show both images side by side as soon as they are uploaded.
    # Each image appears directly below its own uploader.
    if before_file is not None:
        with col_before:
            st.image(Image.open(before_file), caption="Before", use_container_width=True)

    if after_file is not None:
        with col_after:
            st.image(Image.open(after_file), caption="After", use_container_width=True)

    # Step 3: Optional context
    comparison_context = st.text_area(
        label="Additional Context (optional)",
        placeholder="e.g. The After version is a redesign of the checkout flow.",
        height=80,
        key="level2_context",
    )

    # Step 4: Compare button
    compare_clicked = st.button("⚖️ Compare Designs", type="primary", key="compare_btn")

    if compare_clicked:

        # Make sure both files are uploaded before running.
        if before_file is None or after_file is None:
            st.error("Please upload both a Before and an After screenshot.")
        else:
            before_image = Image.open(before_file)
            after_image  = Image.open(after_file)

            with st.spinner("Comparing designs with Gemini Vision..."):
                try:
                    analyzer = GeminiAnalyzer()
                    result = analyzer.compare_designs(
                        before_image=before_image,
                        after_image=after_image,
                        before_filename=before_file.name,
                        after_filename=after_file.name,
                        additional_context=comparison_context or None,
                    )
                    report = result.report
                except Exception as error:
                    st.error(f"Comparison failed: {error}")
                    st.stop()

            st.success("Comparison complete!")

            # ── Scores ────────────────────────────────────────────────────────
            st.subheader("📊 Score Comparison")

            before_score = report.get("before_score", 0)
            after_score  = report.get("after_score", 0)
            score_change = report.get("score_change", 0)

            # Show the three scores in three columns.
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(label="Before Score", value=f"{before_score} / 100")
            sc2.metric(label="After Score",  value=f"{after_score} / 100")

            # The delta on the third metric shows the direction of change.
            if score_change > 0:
                sc3.metric(label="Score Change", value=f"+{score_change}", delta=f"+{score_change} pts")
            elif score_change < 0:
                sc3.metric(label="Score Change", value=str(score_change), delta=f"{score_change} pts")
            else:
                sc3.metric(label="Score Change", value="0", delta="No change")

            # ── Overall verdict ────────────────────────────────────────────────
            verdict = report.get("overall_verdict", "No Significant Change")
            if verdict == "Improved":
                st.success(f"✅ Overall Verdict: {verdict}")
            elif verdict == "Degraded":
                st.error(f"❌ Overall Verdict: {verdict}")
            else:
                st.info(f"➡️ Overall Verdict: {verdict}")

            # ── Summary ────────────────────────────────────────────────────────
            st.subheader("📋 Summary")
            st.write(report.get("summary", "No summary available."))

            # ── Helper to render a list of comparison items ────────────────────
            # Each item has: principle, description, and optionally impact/severity.
            # We reuse this small function for all four lists below.
            def show_items(items, extra_field=None, extra_label=None):
                """Display a list of comparison finding items."""
                if not items:
                    st.write("_None detected._")
                    return
                for item in items:
                    principle   = item.get("principle", "unknown").replace("_", " ").title()
                    description = item.get("description", "No description.")
                    with st.expander(f"**{principle}** — {description[:60]}..."):
                        st.write(f"**Description:** {description}")
                        if extra_field and extra_label:
                            st.write(f"**{extra_label}:** {item.get(extra_field, 'N/A')}")

            # ── Improvements ───────────────────────────────────────────────────
            improvements = report.get("improvements", [])
            st.subheader(f"✅ Improvements ({len(improvements)})")
            show_items(improvements, extra_field="impact", extra_label="Impact")

            # ── Regressions ────────────────────────────────────────────────────
            regressions = report.get("regressions", [])
            st.subheader(f"⚠️ Regressions ({len(regressions)})")
            show_items(regressions, extra_field="impact", extra_label="Impact")

            # ── New Issues ─────────────────────────────────────────────────────
            new_issues = report.get("new_issues", [])
            st.subheader(f"🆕 New Issues Introduced ({len(new_issues)})")
            show_items(new_issues, extra_field="severity", extra_label="Severity")

            # ── Resolved Issues ────────────────────────────────────────────────
            resolved = report.get("resolved_issues", [])
            st.subheader(f"✔️ Resolved Issues ({len(resolved)})")
            show_items(resolved)

            st.caption(f"⏱️ Comparison completed in {result.latency_seconds:.2f} seconds.")