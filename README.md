# Efficient Visual Token Pruning for Vision-Language Question Answering: A Comparative Study of Saliency-Based Reduction in LLaVA

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Course-DIP](https://img.shields.io/badge/Course-Digital%20Image%20Processing-green.svg)](#)

A Digital Image Processing (DIP) mini-project exploring spatial visual token reduction, visual saliency, and Region-of-Interest (ROI) filtering in multimodal Vision-Language Models (VLMs) using the ChartQA benchmark.

---

## Team Information
- **Meet Nachanekar** — Roll No: 23108B0019
- **Arjun Kawale** — Roll No: 23108B0020
- **Department / Class:** EXCS-B

---

## Project Overview
Standard Vision-Language Models (such as LLaVA-1.5-7B) divide high-resolution images into uniform spatial patches (e.g., 14x14 pixels), yielding 576 visual tokens. In document and chart comprehension, substantial visual area is dominated by uniform white background and low-information margins.

This project introduces a **Digital Image Processing and Saliency-Guided Token Pruning Framework** that:
1. Performs low-level image processing, structural analysis, and boundary detection on chart document images.
2. Evaluates patch-level relevance using a hybrid scoring mechanism combining **CLIP visual saliency** and **text-query semantic similarity**.
3. Dynamically discards redundant visual tokens (reducing token budgets to 50%, 25%, and 10%) while maintaining original 2D raster order.
4. Decreases end-to-end LLM sequence length, VRAM consumption, and inference latency with minimal degradation in QA accuracy.

---

## Dataset (Task 1)
- **Source:** HuggingFaceM4/ChartQA
- **Sample Distribution:** 30 curated chart diagrams spanning vertical/horizontal bar graphs, multi-series line plots, and pie charts.
- **Annotations:** Image dimensions, query prompts, and factual ground-truth answers cataloged in `dataset/metadata.csv`.

---

## Task 1 Deliverables
- Project proposal documentation: `docs/Task1_Project_Proposal.pdf`
- Curated sample dataset & metadata: `dataset/`