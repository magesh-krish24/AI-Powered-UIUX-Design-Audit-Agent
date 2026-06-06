"""
app.py - Streamlit UI for the AI-Powered UI/UX Design Audit Agent.

This file handles everything the user sees and interacts with.
It calls the GeminiAnalyzer class to do the actual AI analysis.
"""

import streamlit as st
from PIL import Image

from analyzer import GeminiAnalyzer

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="UI/UX Design Audit Agent", layout="centered")
st.title("🎨 AI-Powered UI/UX Design Audit Agent")
st.write("Upload a screenshot and the AI will audit it for design issues.")

# ── Step 1: File uploader ─────────────────────────────────────────────────────
# Let the user upload a screenshot in any common image format.
uploaded_file = st.file_uploader(
    label="Upload a screenshot",
    type=["png", "jpg", "jpeg", "webp"],
)

# ── Step 2: Optional context ──────────────────────────────────────────────────
# The user can give the AI extra information about the design.
# For example: "This is a mobile checkout flow for first-time buyers."
additional_context = st.text_area(
    label="Additional Context (optional)",
    placeholder="e.g. This is a mobile checkout screen targeting new users.",
    height=80,
)

# ── Step 3: Show the uploaded image ───────────────────────────────────────────
# Display the image so the user can confirm they uploaded the right file.
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", use_container_width=True)

# ── Step 4: Analyze button ────────────────────────────────────────────────────
# When the user clicks this button, we run the AI analysis.
analyze_clicked = st.button("🔍 Analyze Design", type="primary")

if analyze_clicked:

    # Check that the user actually uploaded a file before proceeding.
    if uploaded_file is None:
        st.error("Please upload a screenshot before clicking Analyze.")

    else:
        # Re-open the image in case it was closed after the preview above.
        image = Image.open(uploaded_file)

        # Show a spinner while the AI is working — this can take a few seconds.
        with st.spinner("Analyzing your design with Gemini Vision..."):
            try:
                # Create the analyzer and run the audit.
                analyzer = GeminiAnalyzer()
                result = analyzer.analyze_image(
                    image=image,
                    filename=uploaded_file.name,
                    additional_context=additional_context or None,
                )

                # Pull the report dictionary out of the result object.
                report = result.report

            except Exception as error:
                # If anything goes wrong (bad API key, network issue, etc.)
                # show a friendly error message and stop here.
                st.error(f"Analysis failed: {error}")
                st.stop()

        # ── Step 5: Show results ───────────────────────────────────────────
        st.success("Analysis complete!")

        # ── Screenshot summary ─────────────────────────────────────────────
        st.subheader("📋 Screenshot Summary")
        st.write(report.get("screenshot_summary", "No summary available."))

        # ── Overall score ──────────────────────────────────────────────────
        # Display the overall design quality score out of 100.
        overall_score = report.get("overall_score", 0)
        st.subheader("📊 Overall Design Score")

        # Pick an emoji based on how good the score is.
        if overall_score >= 80:
            score_label = f"✅ {overall_score} / 100 — Good"
        elif overall_score >= 60:
            score_label = f"⚠️ {overall_score} / 100 — Needs Work"
        else:
            score_label = f"❌ {overall_score} / 100 — Poor"

        st.write(score_label)
        st.progress(overall_score / 100)

        # ── Findings ───────────────────────────────────────────────────────
        findings = report.get("findings", [])
        st.subheader(f"🔎 Findings ({len(findings)} issue(s) found)")

        if len(findings) == 0:
            st.write("No issues found. Great design!")

        # Loop through each finding and display it as an expandable card.
        for finding in findings:
            severity   = finding.get("severity", "unknown").upper()
            principle  = finding.get("principle", "unknown").replace("_", " ").title()
            issue      = finding.get("issue", "No description.")
            location   = finding.get("location", "Unknown location")
            impact     = finding.get("user_impact", "No impact described.")
            rec        = finding.get("recommendation", "No recommendation.")
            confidence = finding.get("confidence_score", 0)

            # Choose an emoji based on severity so issues are easy to scan.
            severity_icons = {
                "CRITICAL": "🔴",
                "HIGH":     "🟠",
                "MEDIUM":   "🟡",
                "LOW":      "🔵",
                "INFO":     "⚪",
            }
            icon = severity_icons.get(severity, "❓")

            # Each finding gets its own expander so the page stays tidy.
            with st.expander(f"{icon} [{severity}] {principle} — {location}"):
                st.write(f"**Issue:** {issue}")
                st.write(f"**User Impact:** {impact}")
                st.write(f"**Recommendation:** {rec}")
                st.write(f"**Confidence Score:** {confidence} / 100")

        # ── Latency ────────────────────────────────────────────────────────
        st.caption(f"⏱️ Analysis completed in {result.latency_seconds:.2f} seconds.")
