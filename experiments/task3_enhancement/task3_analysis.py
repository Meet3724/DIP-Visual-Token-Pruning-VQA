import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Import our custom enhancement module
sys.path.append(os.path.abspath(os.path.join("experiments", "task3_enhancement")))
import enhancement as enh

output_dir = os.path.join("outputs", "task3")
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("TASK 3: SPATIAL DOMAIN ENHANCEMENT, HISTOGRAM EQUALIZATION & ARITHMETIC")
print("=" * 80)

# Select 3 diverse sample charts from our dataset
sample_images = ["chart_01.png", "chart_03.png", "chart_05.png"]

# ------------------------------------------------------------------------------
# Tasks 3.2, 3.3, 3.4 & 3.6: Multi-Image Transformation & Histogram Analysis
# ------------------------------------------------------------------------------
for idx, img_name in enumerate(sample_images, start=1):
    img_path = os.path.join("dataset", "images", img_name)
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Skipping.")
        continue
    
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Transformations
    neg = enh.image_negative(rgb)
    log_img = enh.log_transform(rgb)
    gamma_low = enh.gamma_transform(rgb, gamma=0.5)   # Brightens mid-tones
    gamma_high = enh.gamma_transform(rgb, gamma=2.0)  # Darkens mid-tones / increases contrast
    stretched = enh.contrast_stretching(rgb)
    eq_rgb = enh.histogram_equalization(bgr)
    eq_rgb = cv2.cvtColor(eq_rgb, cv2.COLOR_BGR2RGB)
    eq_gray = enh.histogram_equalization(gray)
    
    # Figure 1: Spatial Transformations Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title(f"Original ({img_name})", fontweight="bold"); axes[0, 0].axis("off")
    axes[0, 1].imshow(neg); axes[0, 1].set_title("Image Negative (s = 255 - r)", fontweight="bold"); axes[0, 1].axis("off")
    axes[0, 2].imshow(log_img); axes[0, 2].set_title("Log Transform (s = c*log(1+r))", fontweight="bold"); axes[0, 2].axis("off")
    axes[1, 0].imshow(gamma_low); axes[1, 0].set_title("Power-Law (\u03b3 = 0.5 - Expand Dark)", fontweight="bold"); axes[1, 0].axis("off")
    axes[1, 1].imshow(gamma_high); axes[1, 1].set_title("Power-Law (\u03b3 = 2.0 - Compress Dark)", fontweight="bold"); axes[1, 1].axis("off")
    axes[1, 2].imshow(stretched); axes[1, 2].set_title("Contrast Stretching (Linear)", fontweight="bold"); axes[1, 2].axis("off")
    plt.tight_layout()
    transform_fig_path = os.path.join(output_dir, f"task3_transforms_{img_name.split('.')[0]}.png")
    plt.savefig(transform_fig_path, dpi=250)
    plt.close()
    
    # Figure 2: Histogram Analysis & Equalization
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title(f"Grayscale Original ({img_name})", fontweight="bold"); axes[0, 0].axis("off")
    
    axes[0, 1].hist(gray.ravel(), bins=256, range=[0, 256], color="steelblue", alpha=0.85)
    axes[0, 1].set_title("Original Intensity Histogram", fontweight="bold")
    axes[0, 1].set_xlim([0, 256]); axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].imshow(eq_gray, cmap="gray")
    axes[1, 0].set_title("Histogram Equalized Image", fontweight="bold"); axes[1, 0].axis("off")
    
    axes[1, 1].hist(eq_gray.ravel(), bins=256, range=[0, 256], color="darkred", alpha=0.85)
    axes[1, 1].set_title("Equalized Intensity Histogram", fontweight="bold")
    axes[1, 1].set_xlim([0, 256]); axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    hist_fig_path = os.path.join(output_dir, f"task3_hist_equalization_{img_name.split('.')[0]}.png")
    plt.savefig(hist_fig_path, dpi=250)
    plt.close()
    
    print(f"[{idx}/3] Processed & generated figures for: {img_name}")

# ------------------------------------------------------------------------------
# Task 3.5: Arithmetic Operations
# ------------------------------------------------------------------------------
img_a = cv2.imread(os.path.join("dataset", "images", "chart_01.png"))
img_b = cv2.imread(os.path.join("dataset", "images", "chart_03.png"))

# 1. Addition (Blending two charts)
add_result = enh.image_addition(img_a, img_b, alpha=0.6, beta=0.4)
add_rgb = cv2.cvtColor(add_result, cv2.COLOR_BGR2RGB)

# 2. Subtraction (Background Canvas Isolation / Difference detection)
# Create a synthetic pure white background canvas
white_bg = np.full_like(img_a, 255)
sub_result = enh.image_subtraction(white_bg, img_a)
sub_rgb = cv2.cvtColor(sub_result, cv2.COLOR_BGR2RGB)

# 3. Averaging (Noise suppression across noisy variations)
noise1 = np.clip(img_a.astype(np.int16) + np.random.normal(0, 15, img_a.shape).astype(np.int16), 0, 255).astype(np.uint8)
noise2 = np.clip(img_a.astype(np.int16) + np.random.normal(0, 15, img_a.shape).astype(np.int16), 0, 255).astype(np.uint8)
noise3 = np.clip(img_a.astype(np.int16) + np.random.normal(0, 15, img_a.shape).astype(np.int16), 0, 255).astype(np.uint8)
avg_result = enh.image_averaging([noise1, noise2, noise3])
avg_rgb = cv2.cvtColor(avg_result, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(add_rgb)
axes[0].set_title("Image Addition (Alpha Blending:\nWatermarking / Double Exposure)", fontsize=10, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(sub_rgb)
axes[1].set_title("Image Subtraction (Canvas Diff:\nIsolates Data Marks from White BG)", fontsize=10, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(avg_rgb)
axes[2].set_title("Image Averaging (3-Frame Average:\nAdditive Noise Suppression)", fontsize=10, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
arith_fig_path = os.path.join(output_dir, "task3_arithmetic_operations.png")
plt.savefig(arith_fig_path, dpi=250)
plt.close()
print(f"Generated Arithmetic Operations plot: {arith_fig_path}")

# ------------------------------------------------------------------------------
# Task 3.7: Interactive Menu Implementation
# ------------------------------------------------------------------------------
def interactive_menu():
    print("\n" + "=" * 50)
    print("TASK 3.7: INTERACTIVE IMAGE ENHANCEMENT MENU")
    print("=" * 50)
    img_name = input("Enter image filename from dataset (default: chart_01.png): ").strip() or "chart_01.png"
    filepath = os.path.join("dataset", "images", img_name)
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    img = cv2.imread(filepath)
    print("\nSelect Enhancement Technique:")
    print("1. Image Negative")
    print("2. Log Transformation")
    print("3. Power-Law (Gamma) Transformation")
    print("4. Contrast Stretching")
    print("5. Histogram Equalization")
    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        out = enh.image_negative(img)
        suffix = "negative"
    elif choice == "2":
        out = enh.log_transform(img)
        suffix = "log"
    elif choice == "3":
        g = float(input("Enter gamma value (e.g., 0.5 for bright, 2.0 for dark): ").strip() or "0.5")
        out = enh.gamma_transform(img, gamma=g)
        suffix = f"gamma_{g}"
    elif choice == "4":
        out = enh.contrast_stretching(img)
        suffix = "stretched"
    elif choice == "5":
        out = enh.histogram_equalization(img)
        suffix = "hist_eq"
    else:
        print("Invalid selection.")
        return

    save_path = os.path.join(output_dir, f"interactive_{suffix}_{img_name}")
    cv2.imwrite(save_path, out)
    print(f"Enhanced image saved successfully to: {save_path}")

# Run non-interactively if called in automation, but keep method available
if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_menu()
else:
    print("\nBatch pipeline finished. To run the Task 3.7 interactive menu anytime, run:")
    print("python experiments\\task3_enhancement\\task3_analysis.py --interactive")

print("=" * 80)
