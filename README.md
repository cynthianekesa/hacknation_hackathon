# 🏥 NaviCare: Agentic Healthcare Intelligence for Rural India

> **Trust-scored healthcare discovery for 1.4 billion Indians. Turning 10,000 messy facility records into a living intelligence network.**

[![Hack-Nation 2026](https://img.shields.io/badge/Hack--Nation-2026-blue)](https://hack-nation.ai)
[![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-red)](https://databricks.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.0-green)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Table of Contents

- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Architecture](#-architecture)
- [Key Results](#-key-results)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [MLflow Traces](#-mlflow-traces)
- [Self-Correction Example](#-self-correction-example)
- [Medical Desert Maps](#-medical-desert-maps)
- [Challenge Requirements](#-challenge-requirements)
- [Target Audience](#-target-audience)
- [Demo Script](#-demo-script)
- [Future Work](#-future-work)
- [Acknowledgments](#-acknowledgments)

---

## 📌 The Problem

### In India, a postal code determines a lifespan.

**70% of India's 1.4 billion people live in rural areas**, yet healthcare access remains a discovery crisis. Families travel hours only to find a facility lacks the specific ICU bed, oxygen supply, or specialist they urgently need.

### The issue isn't a lack of hospitals—it's a lack of VERIFIABLE intelligence.

| Problem | Reality |
|---------|---------|
| **Fragmented Data** | Facility records are scattered across Excel sheets, PDFs, and handwritten notes |
| **Unstructured Format** | Critical info buried in free-text descriptions with no standardization |
| **Contradictions** | "Advanced surgery available" but "No anesthesiologist on staff" |
| **No Verification** | No way to know if a facility's claims are actually true |
| **No Discovery** | Can't search "Which rural hospital has an ICU right now?" |

### The Human Cost

A mother in rural Bihar with a sick child:
- Drives 2 hours to the nearest hospital
- Finds out they have no pediatric ICU
- Drives another hour to the next facility
- Wastes critical hours that could save her child's life

**We built NaviCare so no family ever has to guess where to find help again.**

---

## 🎯 Our Solution

**NaviCare** transforms 10,000+ messy healthcare facility records into a verifiable, searchable intelligence network with:

### Core Features

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | **Unstructured Extraction** | Parses ICU, surgery, emergency capabilities from free-text notes using pattern matching + Pydantic models | ✅ Complete |
| 2 | **Multi-Attribute Reasoning** | Answers complex queries like "Find rural Bihar facilities with emergency appendectomy AND part-time doctors" | ✅ Complete |
| 3 | **Trust Scorer** | Statistical confidence scores with 95% bootstrap confidence intervals | ✅ Complete |
| 4 | **Contradiction Detection** | Flags "surgery without anesthesiologist", "24/7 with part-time doctors", and more | ✅ Complete |
| 5 | **Validator Agent** | Self-correcting loop that cross-references medical standards and demotes low-trust claims | ✅ Complete |
| 6 | **Medical Desert Mapper** | Identifies regional gaps by pincode with actionable insights for NGO planners | ✅ Complete |
| 7 | **MLflow 3 Tracing** | Full transparency—every decision traced to source sentence with span-level attributes | ✅ Complete |
| 8 | **Tavily Web Validation** | Real-time external verification of facility claims via web search | 🚧 Architecture ready* |

> *Due to Databricks Free Trial network restrictions, Tavily integration is architecturally complete with graceful fallback. Ready to deploy with production network access.

### What Makes NaviCare Different

| Feature | Typical Solution | NaviCare |
|---------|-----------------|----------|
| Output | "Facility has ICU" | "87% confidence (95% CI: 82-92%)" |
| Contradictions | Ignores them | Flags + demotes trust score |
| Traceability | None | Every decision traced to source sentence |
| Error handling | Fails silently | Self-corrects via Validator Agent |
| Medical deserts | Binary (has/doesn't have) | Risk scores with confidence intervals |

---

## 🏗️ Architecture
