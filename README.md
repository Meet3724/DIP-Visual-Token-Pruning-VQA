# Efficient Visual Token Pruning for Vision-Language Question Answering

> **A Saliency-Guided Digital Image Processing Pipeline for Token Reduction in LLaVA-1.5 on ChartQA**  
> *Course:* Digital Image Processing (DIP) Mini-Project  
> *Class / Branch:* EXCS-B  
> *Academic Year:* 2026  

---

## Team Information

| Name | Roll Number | Class / Branch | Primary Responsibilities |
| :--- | :---: | :---: | :--- |
| **Meet Nachanekar** | `23108B0019` | EXCS-B | Pipeline Architecture, Spatial Filtering, Region Growing & Morphological Benchmarking |
| **Arjun Kawale** | `23108B0020` | EXCS-B | Image Enhancement, Edge Operators, Quadtree Decomposition & Dataset Curation |

---

## Project Overview

Large Vision-Language Models (such as **LLaVA-1.5-7B**) convert high-resolution image inputs into a spatial sequence of $14 \times 14$ non-overlapping patches using a CLIP ViT-L/14 vision backbone, generating **576 visual tokens** per image. In statistical document graphics (bar charts, line plots, and categorical diagrams), **$60\%$ to $78\%$ of the visual area is dominated by uninformative background canvas (whitespace)**.

This project implements an end-to-end **Digital Image Processing (DIP)** pipeline designed to extract structural features, segment graphical foreground objects, and evaluate region saliency. By combining deterministic classical operators with patch-level density metrics, the framework identifies and discards redundant background canvas tokens prior to Large Language Model (LLM) self-attention, reducing sequence lengths, memory overhead, and inference latency with minimal loss in Question Answering fidelity.

---

## Semester Milestone Roadmap (Tasks 1–8)

| Task | Module | Core DIP Techniques | Status |
| :---: | :--- | :--- | :---: |
| **Task 1** | **Project Proposal & Dataset Setup** | Problem formulation, architectural design, ChartQA benchmark curation ($30$ samples) | **Completed** |
| **Task 2** | **Image Fundamentals & Sampling** | Matrix indexing, RGB/Grayscale/HSV spaces, spatial decimation, quantization, lossless PNG vs. lossy JPEG | **Completed** |
| **Task 3** | **Spatial Domain Enhancement** | Intensity point transforms (Log, Gamma $\gamma=0.5, 2.0$, Negative), piecewise Contrast Stretching, Histogram Equalization, canvas subtraction | **Completed** |
| **Task 4** | **Spatial Domain Filtering (Smoothing)** | Impulse (Salt-and-Pepper) & Gaussian noise modeling, Linear Mean ($3\times3, 5\times5$), Gaussian, Non-linear Median ($3\times3, 5\times5$), PSNR/SSIM evaluation | **Completed** |
| **Task 5** | **Sharpening & Integrated Pipeline** | Laplacian ($\nabla^2$), Sobel gradient magnitude, High-Boost ($A=1.2$), 3-stage integration (**Acquisition $\to$ Enhancement $\to$ Filtering**) | **Completed** |
| **Task 6** | **Segmentation: Edges & Thresholding** | Roberts, Prewitt, Sobel, Laplacian, Global ($T=210$), Otsu's optimal thresholding, Adaptive Gaussian breakdown, foreground density metrics | **Completed** |
| **Task 7** | **Region-Based Segmentation** | Seeded Region Growing (queue BFS, dynamic $\mu_R$, $T=10, 25, 65$), 4- vs. 8-connectivity evaluation, Quadtree Split-and-Merge ($730$ leaf blocks) | **Completed** |
| **Task 8** | **Binary Morphological Processing** | Inverted threshold binarization, Erosion ($A \ominus B$) & Dilation ($A \oplus B$) with $3\times3$ Rect/Cross and $5\times5$ Rect SEs, Morphological Boundary $\beta(A)$, iterative degradation ($k=1,2,3$) | **Completed** |

---

## Experimental Findings & Key Insights

### 1. File Formats & Fundamentals (Task 2)
* **Compression Efficiency:** On synthetic document images, lossless **PNG ($36.75\text{ KB}$)** is **$49.4\%$ smaller than JPEG ($72.69\text{ KB}$)**. PNG’s DEFLATE algorithm encodes large contiguous canvas regions at near-zero entropy, whereas JPEG’s $8 \times 8$ block DCT introduces high-frequency ringing artifacts along sharp text boundaries.
* **Spatial Decimation:** Naive spatial downsampling ($25\%$) leads to severe aliasing, destroying numerical tick marks and decimal values, establishing the necessity of **full-resolution selective patch pruning** over global image resizing.

### 2. Spatial Enhancement & Arithmetic (Task 3)
* **Histogram Characteristics:** Chart images exhibit highly multimodal discrete histograms dominated by a massive white spike at $r = 255$. Global histogram equalization redistributes this peak, converting the clean background into gray noise. **Linear contrast stretching** preserves canvas integrity while expanding data bar contrast.
* **Arithmetic Separation:** Background canvas subtraction ($g = \vert{}f_{\text{canvas}} - f_{\text{chart}}\vert{}$) achieves zero-overhead foreground isolation.

### 3. Noise Restoration & Filtering (Tasks 4 & 5)
* **Impulse Noise Suppression:** For salt-and-pepper corruption, the **$3 \times 3$ Non-Linear Median Filter** achieved superior performance (**$25.93\text{ dB}$ PSNR, $0.9520$ SSIM** on `chart_01.png`), eliminating outliers while preserving sharp step edges. Linear Mean and Gaussian filters smeared text into unreadable gray halos.
* **Sharpening & High-Boost:** Sobel gradient magnitude vectors and High-Boost filtering ($A = 1.15$) restore boundary gradients across data bars and grid lines. The textbook formula ($g = A \cdot f - \bar{f}$) shifts the continuous white canvas DC level, requiring parameter tuning to prevent canvas darkening.

### 4. Edge Detection & Thresholding (Task 6)
* **Segmentation Benchmark:** Otsu's thresholding automatically determined optimal separation thresholds ($T_{\text{opt}} \in [181.0, 198.0]$), separating colored bars and alphanumeric annotations from whitespace.
* **Failure of Adaptive Thresholding:** Adaptive Gaussian thresholding failed catastrophically on uniform canvas regions, generating extensive false-positive noise due to local micro-variations falling below the neighborhood mean.
* **Pruning Predictor:** Foreground density calculations confirmed that **$60.39\%$ to $77.75\%$ of the total pixel canvas contains zero data**, providing empirical validation for dropping up to $60\%$ of background tokens.

### 5. Region-Based & Morphological Operations (Tasks 7 & 8)
* **Neighborhood Connectivity:** In region growing, 8-connectivity is essential for chart diagrams to preserve diagonal text strokes, slanted trend lines, and rotated axis labels, avoiding the artificial disconnections caused by 4-connectivity.
* **Quadtree Decomposition:** Region split-and-merge generated **730 leaf blocks** on sample crops. Uniform canvas blocks remained as large unified quadrants, while information-dense text and bar junctions recursively subdivided to minimum leaf limits ($8 \times 8$).
* **Morphological Bounds:** A single iteration ($k = 1$) of dilation using a $3 \times 3$ cross SE bridges broken font characters and internal pinholes (+12.04% area), while erosion ($k = 1$) strips isolated single-pixel noise (-10.86% area). Repeated erosion ($k = 3$) strips thin axis numbers (-27.14% area), whereas repeated dilation ($k = 3$) fuses adjacent bars together (+41.64% area).

---

## Repository Architecture

```text
DIP-Visual-Token-Pruning-VQA/
├── dataset/
│   ├── images/                           # Curated ChartQA sample diagrams (chart_01 to chart_30)
│   └── metadata.csv                      # Query prompts, QA ground truth, and image properties
├── docs/
│   ├── Task1_Project_Proposal.pdf        # Project scope and milestone roadmap
│   ├── Task2_Report.pdf                  # Fundamentals, quantization & format benchmarks
│   ├── Task3_Report.pdf                  # Spatial intensity transformations & arithmetic
│   ├── Task4_Report.pdf                  # Noise simulation, smoothing & restoration
│   ├── Task5_Report.pdf                  # Sharpening, multi-image comparison & 3-stage pipeline
│   ├── Task6_Report.pdf                  # Edge detection, thresholding & density analysis
│   ├── Task7_Report.pdf                  # Region growing, connectivity & quadtree split-merge
│   └── Task8_Report.pdf                  # Binary morphology, erosion, dilation & boundaries
├── experiments/
│   ├── task2_fundamentals/
│   │   └── task2_analysis.py             # Matrix indexing, color spaces & sampling
│   ├── task3_enhancement/
│   │   ├── enhancement.py                # Point transforms, histogram equalization, arithmetic
│   │   └── task3_analysis.py             # Multi-image enhancement & interactive CLI
│   ├── task4_filtering/
│   │   ├── filtering.py                  # Noise injection, Mean, Gaussian, Median filters
│   │   └── task4_analysis.py             # Denoising benchmarks (PSNR / SSIM)
│   ├── task5_pipeline/
│   │   └── task5_analysis.py             # Gradient sharpening & 3-stage integrated pipeline
│   ├── task6_segmentation/
│   │   └── task6_analysis.py             # Edge operators (Roberts/Prewitt/Sobel) & Otsu/Adaptive
│   ├── task7_region_segmentation/
│   │   └── task7_analysis.py             # BFS region growing, 4/8-conn & Quadtree split-merge
│   └── task8_morphology/
│       └── task8_analysis.py             # Binary erosion, dilation, SEs & iterative morphology
├── outputs/
│   ├── task2/ ... task8/                 # Generated visual matrices, comparison plots, and metrics
├── requirements.txt                      # Complete Python environment dependencies
└── README.md                             # Master project documentation
```

---

## Installation & Setup

### 1. Clone the Repository

```
git clone [https://github.com/Meet3724/DIP-Visual-Token-Pruning-VQA.git](https://github.com/Meet3724/DIP-Visual-Token-Pruning-VQA.git)
cd DIP-Visual-Token-Pruning-VQA
```

### 2. Configure Virtual Environment

```
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running Experiments
    
Run the batch benchmarks or interactive interfaces using the commands below:

```
# Task 2: Image Representation, Sampling & Formats
python experiments/task2_fundamentals/task2_analysis.py

# Task 3: Spatial Enhancement, Histograms & Arithmetic
python experiments/task3_enhancement/task3_analysis.py
python experiments/task3_enhancement/task3_analysis.py --interactive

# Task 4: Spatial Filtering & Denoising Benchmark
python experiments/task4_filtering/task4_analysis.py

# Task 5: Sharpening, Multi-Filter Comparison & 3-Stage Pipeline
python experiments/task5_pipeline/task5_analysis.py
python experiments/task5_pipeline/task5_analysis.py --interactive

# Task 6: Edge Detection, Thresholding & Foreground Density
python experiments/task6_segmentation/task6_analysis.py
python experiments/task6_segmentation/task6_analysis.py --interactive

# Task 7: Region Growing, Connectivity & Quadtree Split-and-Merge
python experiments/task7_region_segmentation/task7_analysis.py
python experiments/task7_region_segmentation/task7_analysis.py --interactive

# Task 8: Morphological Erosion, Dilation & Iterative Processing
python experiments/task8_morphology/task8_analysis.py
python experiments/task8_morphology/task8_analysis.py --interactive
```

