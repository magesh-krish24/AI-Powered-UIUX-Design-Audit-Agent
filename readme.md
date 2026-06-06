# AI-Powered UI/UX Design Audit Agent

## Overview

AI-Powered UI/UX Design Audit Agent is a computer vision-based application that analyzes website and application screenshots using Google's Gemini Vision model.

The system evaluates designs based on key UI/UX principles and generates a structured audit report containing design issues, severity levels, confidence scores, and recommendations.

This project was developed as part of an AI/ML Engineering Hiring Challenge.

---

## Features

* Upload website or application screenshots
* AI-powered screenshot analysis
* Automatic screenshot summary generation
* Visual Hierarchy evaluation
* Contrast analysis (WCAG-inspired)
* Spacing analysis
* Alignment analysis
* Consistency analysis
* Severity classification
* Confidence scoring
* Overall Design Score
* Structured audit report

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### AI Model

* Google Gemini Vision (Gemini 2.5 Flash)

### Libraries

* google-generativeai
* pillow
* python-dotenv
* streamlit

---

## Project Structure

```text
UIUX-Design-Audit-Agent
│
├── app.py
├── analyzer.py
├── config.py
├── prompts.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

#### app.py

Handles the Streamlit user interface, image upload, and displaying results.

#### analyzer.py

Processes screenshots and communicates with Gemini Vision for design analysis.

#### prompts.py

Contains prompt engineering logic and analysis instructions.

#### config.py

Stores application configuration and environment variable handling.

---

## Workflow

```text
User Uploads Screenshot
        ↓
Streamlit UI
        ↓
Image Validation
        ↓
Prompt Generation
        ↓
Gemini Vision Analysis
        ↓
JSON Response
        ↓
Score Calculation
        ↓
Results Display
```

---

## Design Principles Evaluated

### 1. Visual Hierarchy

Checks whether important elements attract attention appropriately.

### 2. Contrast

Evaluates readability and accessibility of interface elements.

### 3. Spacing

Identifies overcrowded or inconsistent spacing patterns.

### 4. Alignment

Ensures elements are positioned consistently.

### 5. Consistency

Checks whether components follow a unified design language.

---

## Sample Output

The system generates:

* Screenshot Summary
* Overall Design Score
* Severity Classification
* Confidence Scores
* Detailed Findings
* Improvement Recommendations

---

## Installation

### Clone Repository

```bash
git clone https://github.com/magesh-krish24/AI-Powered-UIUX-Design-Audit-Agent.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Environment File

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### Run Application

```bash
streamlit run app.py
```

---

## Challenges Faced

* Gemini API configuration
* Model compatibility issues
* Environment variable management
* Structured JSON response parsing
* Prompt engineering for consistent design audits

---

## Future Enhancements

### Level 2

* Before vs After Screenshot Comparison
* Regression Detection
* Improvement Classification
* UI Change Tracking

### Level 3

* Multi-screen Audit Workflows
* Automated UX Recommendations
* PDF Report Export
* Historical Audit Tracking

---

## Author

**Magesh S**

Master of Computer Applications (MCA)

Kumaraguru College of Technology

AI/ML Engineering Hiring Challenge Submission
