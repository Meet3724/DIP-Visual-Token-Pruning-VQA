import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Connect to previous filtering & enhancement modules
sys.path.append(os.path.abspath(os.path.join("experiments", "task3_enhancement")))
sys.path.append(os.path.abspath(os.path.join("experiments", "task4_filtering")))
import enhancement as enh
import filtering as filt

output_dir = os.path.join("outputs", "task6")
os.makedirs(output_dir, exist_ok=True)

print("=" * 95)
print("TASK 6: IMAGE SEGMENTATION - EDGE DETECTION AND THRESHOLDING")
print("=" * 95)

sample_images = ["chart_01.png", "chart_03.png", "chart_05.png"]

# -------------------------------------------------------------------------
# Edge Detection Operators
# -------------------------------------------------------------------------

def roberts_edge_detector(gray_img):
    """Computes Roberts Cross Edge Detection (2x2 diagonal kernels)"""
    kernel_x = np.array([[ 1,  0],
                         [ 0, -1]], dtype=np.float32)
    kernel_y = np.array([[ 0,  1],
                         [-1,  0]], dtype=np.float32)
    gx = cv2.filter2D(gray_img.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(gray_img.astype(np.float32), -1, kernel_y)
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def prewitt_edge_detector(gray_img):
    """Computes Prewitt Edge Detection (3x3 uniform directional kernels)"""
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]], dtype=np.float32)
    gx = cv2.filter2D(gray_img.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(gray_img.astype(np.float32), -1, kernel_y)
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def sobel_edge_detector(gray_img):
    """Computes Sobel Edge Detection (3x3 weighted directional kernels)"""
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def laplacian_edge_detector(gray_img):
    """Computes Second-Order Laplacian Derivative (isotropic edge detector)"""
    lap = cv2.Laplacian(gray_img, cv2.CV_32F, ksize=3)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)

# -------------------------------------------------------------------------
# Thresholding Segmentation Operators
# (Note: Inverted binary THRESH_BINARY_INV makes foreground data marks white (255)
#  and background whitespace black (0) for clear object segmentation)
# -------------------------------------------------------------------------

def global_threshold(gray_img, thresh=210):
    """Fixed Global Thresholding: separates foreground data marks from white canvas"""
    _, binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary

def otsu_threshold(gray_img):
    """Automatic Otsu's Thresholding: calculates optimal T by maximizing between-class variance"""
    optimal_t, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary, optimal_t

def adaptive_threshold(gray_img, block_size=15, C=4):
    """Adaptive Gaussian Thresholding: computes local thresholds over non-uniform neighborhoods"""
    return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block_size, C)

# -------------------------------------------------------------------------
# Batch Analysis across 3 Chart Images
# -------------------------------------------------------------------------

otsu_threshold_values = {}

for idx, img_name in enumerate(sample_images, start=1):
    img_path = os.path.join("dataset", "images", img_name)
    if not os.path.exists(img_path):
        continue

    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prefix = img_name.split('.')[0]

    # Task 6.1: Prepare Input Image using Task 5 Best Pipeline (Contrast Stretched + Median Denoised)
    enhanced = enh.contrast_stretching(rgb)
    denoised = filt.median_filter(enhanced, ksize=3)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)

    # Task 6.2: Edge Detection Implementations
    edge_roberts = roberts_edge_detector(gray)
    edge_prewitt = prewitt_edge_detector(gray)
    edge_sobel   = sobel_edge_detector(gray)
    edge_laplace = laplacian_edge_detector(gray)

    # Task 6.3: Thresholding Implementations
    thresh_global = global_threshold(gray, thresh=210)
    thresh_otsu, opt_t = otsu_threshold(gray)
    thresh_adapt  = adaptive_threshold(gray, block_size=15, C=4)
    otsu_threshold_values[img_name] = opt_t

    # Save individual edge and threshold assets
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_edge_roberts.png"), edge_roberts)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_edge_prewitt.png"), edge_prewitt)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_edge_sobel.png"), edge_sobel)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_edge_laplacian.png"), edge_laplace)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_thresh_global.png"), thresh_global)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_thresh_otsu.png"), thresh_otsu)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_thresh_adaptive.png"), thresh_adapt)

    # Figure 1: Edge Detection Comparison Grid
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    axes[0].imshow(rgb); axes[0].set_title(f"Preprocessed Input ({img_name})", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(edge_roberts, cmap="gray"); axes[1].set_title("Roberts Operator (2x2)", fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(edge_prewitt, cmap="gray"); axes[2].set_title("Prewitt Operator (3x3)", fontweight="bold"); axes[2].axis("off")
    axes[3].imshow(edge_sobel, cmap="gray"); axes[3].set_title("Sobel Operator (3x3 - Optimal)", fontweight="bold"); axes[3].axis("off")
    axes[4].imshow(edge_laplace, cmap="gray"); axes[4].set_title("Laplacian (\u2207\u00b2)", fontweight="bold"); axes[4].axis("off")
    plt.tight_layout()
    edge_fig_path = os.path.join(output_dir, f"task6_edge_comparison_{prefix}.png")
    plt.savefig(edge_fig_path, dpi=250)
    plt.close()

    # Figure 2: Thresholding Segmentation Comparison Grid
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title(f"Grayscale Input ({img_name})", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(thresh_global, cmap="gray"); axes[1].set_title("Global Threshold (T = 210)", fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(thresh_otsu, cmap="gray"); axes[2].set_title(f"Otsu's Threshold (T = {opt_t:.1f} - Optimal)", fontweight="bold"); axes[2].axis("off")
    axes[3].imshow(thresh_adapt, cmap="gray"); axes[3].set_title("Adaptive Gaussian Threshold", fontweight="bold"); axes[3].axis("off")
    plt.tight_layout()
    thresh_fig_path = os.path.join(output_dir, f"task6_threshold_comparison_{prefix}.png")
    plt.savefig(thresh_fig_path, dpi=250)
    plt.close()

    print(f"[{idx}/3] Processed & generated figures for: {img_name} (Otsu Optimal T = {opt_t:.1f})")

# -------------------------------------------------------------------------
# Quantitative Foreground Density Analysis (DIP Token Saliency Metric)
# -------------------------------------------------------------------------
print("\n" + "=" * 95)
print("SEGMENTATION FOREGROUND RATIO (DIP TOKEN PRUNING PREDICTOR)")
print(f"{'Image':<14} | {'Method':<20} | {'Foreground Px':<15} | {'Canvas Ratio (%)':<18} | {'DIP Observation'}")
print("-" * 95)
for img_name in sample_images:
    prefix = img_name.split('.')[0]
    otsu_path = os.path.join(output_dir, f"{prefix}_thresh_otsu.png")
    sobel_path = os.path.join(output_dir, f"{prefix}_edge_sobel.png")
    if os.path.exists(otsu_path) and os.path.exists(sobel_path):
        o_img = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)
        s_img = cv2.imread(sobel_path, cv2.IMREAD_GRAYSCALE)
        tot_px = o_img.shape[0] * o_img.shape[1]

        o_fg = np.sum(o_img > 0)
        s_fg = np.sum(s_img > 40)

        o_ratio = (o_fg / tot_px) * 100.0
        s_ratio = (s_fg / tot_px) * 100.0

        print(f"{img_name:<14} | {'Otsu Mask':<20} | {o_fg:<15,} | {o_ratio:<18.2f} | Direct data-region density")
        print(f"{img_name:<14} | {'Sobel Edge Map':<20} | {s_fg:<15,} | {s_ratio:<18.2f} | High-frequency contour density")
print("=" * 95)

# -------------------------------------------------------------------------
# Interactive CLI Runner (Task 6 Demonstration Tool)
# -------------------------------------------------------------------------
def interactive_segmentation():
    print("\n" + "=" * 55)
    print("TASK 6: INTERACTIVE SEGMENTATION DEMO")
    print("=" * 55)
    img_name = input("Enter image filename from dataset (default: chart_01.png): ").strip() or "chart_01.png"
    filepath = os.path.join("dataset", "images", img_name)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    bgr = cv2.imread(filepath)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    print("\nSelect Segmentation Technique:")
    print("1. Roberts Edge Detection")
    print("2. Prewitt Edge Detection")
    print("3. Sobel Edge Detection")
    print("4. Laplacian Operator")
    print("5. Global Thresholding")
    print("6. Otsu's Automatic Thresholding")
    print("7. Adaptive Gaussian Thresholding")
    choice = input("Enter choice (1-7): ").strip()

    if choice == "1":
        res = roberts_edge_detector(gray); suffix = "roberts"
    elif choice == "2":
        res = prewitt_edge_detector(gray); suffix = "prewitt"
    elif choice == "3":
        res = sobel_edge_detector(gray); suffix = "sobel"
    elif choice == "4":
        res = laplacian_edge_detector(gray); suffix = "laplacian"
    elif choice == "5":
        t = int(input("Enter threshold value (0-255, default 210): ").strip() or "210")
        res = global_threshold(gray, thresh=t); suffix = f"global_T{t}"
    elif choice == "6":
        res, opt_t = otsu_threshold(gray); suffix = f"otsu_T{int(opt_t)}"
        print(f"Calculated Otsu's optimal threshold: {opt_t:.2f}")
    elif choice == "7":
        res = adaptive_threshold(gray); suffix = "adaptive"
    else:
        print("Invalid selection.")
        return

    out_file = os.path.join(output_dir, f"interactive_{suffix}_{img_name}")
    cv2.imwrite(out_file, res)
    print(f"Segmentation result successfully saved to: {out_file}")

if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_segmentation()
else:
    print("\nBatch analysis completed. Run the interactive CLI live anytime with:")
    print("python experiments\\task6_segmentation\\task6_analysis.py --interactive")
print("=" * 95)
