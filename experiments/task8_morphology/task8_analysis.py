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

output_dir = os.path.join("outputs", "task8")
os.makedirs(output_dir, exist_ok=True)

print("=" * 95)
print("TASK 8: BINARY MORPHOLOGICAL PROCESSING - EROSION AND DILATION")
print("=" * 95)

# -------------------------------------------------------------------------
# Task 8.1: Convert Segmented Image into a Binary Image
# -------------------------------------------------------------------------
img_name = "chart_01.png"
img_path = os.path.join("dataset", "images", img_name)

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Input image not found at: {img_path}")

bgr_img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# Preprocessing from Task 5 (Median denoising + contrast stabilization)
denoised_gray = filt.median_filter(gray_img, ksize=3)

# Experiment with multiple thresholding values (T1 = 150, T2 = Otsu, T3 = 220)
# Use THRESH_BINARY_INV so foreground objects (bars, text) = 255 (White), background = 0 (Black)
t_val1 = 150
_, bin_t1 = cv2.threshold(denoised_gray, t_val1, 255, cv2.THRESH_BINARY_INV)

otsu_val, bin_otsu = cv2.threshold(denoised_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

t_val3 = 220
_, bin_t3 = cv2.threshold(denoised_gray, t_val3, 255, cv2.THRESH_BINARY_INV)

# Primary binary image used for morphology
binary_base = bin_otsu
total_pixels = binary_base.shape[0] * binary_base.shape[1]
base_fg_pixels = int(np.sum(binary_base == 255))

print(f"\n[Task 8.1] Binarization Comparison for '{img_name}':")
print(f"  • Threshold T = {t_val1:<3}        : {np.sum(bin_t1 == 255):<10,} foreground px | Clips light pastel bars")
print(f"  • Otsu Optimal T = {otsu_val:<5.1f} : {base_fg_pixels:<10,} foreground px | Optimal foreground separation")
print(f"  • Threshold T = {t_val3:<3}        : {np.sum(bin_t3 == 255):<10,} foreground px | Captures faint gridlines and text halos")

# Figure 1: Binarization Experiment Grid
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(rgb_img); axes[0].set_title(f"Original RGB ({img_name})", fontweight="bold"); axes[0].axis("off")
axes[1].imshow(bin_t1, cmap="gray"); axes[1].set_title(f"Binary T = {t_val1} (Strict)", fontweight="bold"); axes[1].axis("off")
axes[2].imshow(bin_otsu, cmap="gray"); axes[2].set_title(f"Binary Otsu T = {otsu_val:.1f} (Optimal)", fontweight="bold"); axes[2].axis("off")
axes[3].imshow(bin_t3, cmap="gray"); axes[3].set_title(f"Binary T = {t_val3} (Permissive)", fontweight="bold"); axes[3].axis("off")
plt.tight_layout()
fig1_path = os.path.join(output_dir, "task8_binarization_comparison.png")
plt.savefig(fig1_path, dpi=250)
plt.close()
print(f"  -> Saved Binarization comparison plot to: {fig1_path}")

# -------------------------------------------------------------------------
# Structuring Elements Definition (Tasks 8.2 & 8.3)
# -------------------------------------------------------------------------
se_3x3_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
se_5x5_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

structuring_elements = [
    ("3x3 Square SE", se_3x3_rect),
    ("Cross-Shaped SE (3x3)", se_cross),
    ("5x5 Square SE", se_5x5_rect)
]

# -------------------------------------------------------------------------
# Task 8.2: Implement Erosion
# -------------------------------------------------------------------------
erosion_results = []
for name, se in structuring_elements:
    eroded = cv2.erode(binary_base, se, iterations=1)
    fg_px = int(np.sum(eroded == 255))
    area_change = ((fg_px - base_fg_pixels) / base_fg_pixels) * 100.0
    erosion_results.append((name, eroded, fg_px, area_change))

# -------------------------------------------------------------------------
# Task 8.3: Implement Dilation
# -------------------------------------------------------------------------
dilation_results = []
for name, se in structuring_elements:
    dilated = cv2.dilate(binary_base, se, iterations=1)
    fg_px = int(np.sum(dilated == 255))
    area_change = ((fg_px - base_fg_pixels) / base_fg_pixels) * 100.0
    dilation_results.append((name, dilated, fg_px, area_change))

# Figure 2: Structuring Elements Effect on Erosion and Dilation
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes[0, 0].imshow(binary_base, cmap="gray"); axes[0, 0].set_title("Input Binary (Otsu)", fontweight="bold"); axes[0, 0].axis("off")
axes[1, 0].imshow(binary_base, cmap="gray"); axes[1, 0].set_title("Input Binary (Otsu)", fontweight="bold"); axes[1, 0].axis("off")

for i, (name, eroded, fg_px, delta) in enumerate(erosion_results, start=1):
    axes[0, i].imshow(eroded, cmap="gray")
    axes[0, i].set_title(f"Erosion: {name}\n{fg_px:,} px ({delta:+.1f}%)", fontweight="bold")
    axes[0, i].axis("off")

for i, (name, dilated, fg_px, delta) in enumerate(dilation_results, start=1):
    axes[1, i].imshow(dilated, cmap="gray")
    axes[1, i].set_title(f"Dilation: {name}\n{fg_px:,} px ({delta:+.1f}%)", fontweight="bold")
    axes[1, i].axis("off")

plt.tight_layout()
fig2_path = os.path.join(output_dir, "task8_se_comparison.png")
plt.savefig(fig2_path, dpi=250)
plt.close()
print(f"  -> Saved Structuring Elements Comparison plot to: {fig2_path}")

# Print Quantitative SE Table
print("\n" + "-" * 95)
print(f"{'Operation':<10} | {'Structuring Element':<24} | {'Foreground Px':<15} | {'Area Change (%)':<16} | {'DIP Observation'}")
print("-" * 95)
for name, _, fg_px, delta in erosion_results:
    obs = "Mild contraction; strips single-pixel noise" if "3x3" in name and "Cross" not in name else (
          "Directional contraction; preserves diagonal connectivity" if "Cross" in name else "Heavy boundary shrinkage; text breaks")
    print(f"{'Erosion':<10} | {name:<24} | {fg_px:<15,} | {delta:<+16.2f} | {obs}")
for name, _, fg_px, delta in dilation_results:
    obs = "Uniform expansion; closes text gaps" if "3x3" in name and "Cross" not in name else (
          "Cross expansion; connects orthogonal strokes" if "Cross" in name else "Aggressive expansion; merges adjacent bars")
    print(f"{'Dilation':<10} | {name:<24} | {fg_px:<15,} | {delta:<+16.2f} | {obs}")
print("-" * 95)

# -------------------------------------------------------------------------
# Task 8.4: Comparison & Morphological Boundary Detection
# -------------------------------------------------------------------------
# Morphological Internal Boundary: beta(A) = A - (A (-) B)
# Morphological Gradient: G = (A (+) B) - (A (-) B)
se_eval = se_3x3_rect
eroded_eval = cv2.erode(binary_base, se_eval, iterations=1)
dilated_eval = cv2.dilate(binary_base, se_eval, iterations=1)

internal_boundary = cv2.subtract(binary_base, eroded_eval)
morph_gradient = cv2.subtract(dilated_eval, eroded_eval)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(binary_base, cmap="gray"); axes[0].set_title("Binary Base Object A", fontweight="bold"); axes[0].axis("off")
axes[1].imshow(eroded_eval, cmap="gray"); axes[1].set_title("Eroded A \u2296 B (Contraction)", fontweight="bold"); axes[1].axis("off")
axes[2].imshow(dilated_eval, cmap="gray"); axes[2].set_title("Dilated A \u2295 B (Expansion)", fontweight="bold"); axes[2].axis("off")
axes[3].imshow(internal_boundary, cmap="gray"); axes[3].set_title("Internal Boundary A - (A \u2296 B)", fontweight="bold"); axes[3].axis("off")
plt.tight_layout()
fig3_path = os.path.join(output_dir, "task8_morphology_comparison_boundary.png")
plt.savefig(fig3_path, dpi=250)
plt.close()
print(f"  -> Saved Morphological Boundary plot to: {fig3_path}")

# -------------------------------------------------------------------------
# Task 8.5: Experiment with Number of Iterations (1, 2, 3)
# -------------------------------------------------------------------------
iterations_list = [1, 2, 3]
multi_erosion = []
multi_dilation = []

for it in iterations_list:
    e_it = cv2.erode(binary_base, se_3x3_rect, iterations=it)
    d_it = cv2.dilate(binary_base, se_3x3_rect, iterations=it)
    e_px = int(np.sum(e_it == 255))
    d_px = int(np.sum(d_it == 255))
    multi_erosion.append((it, e_it, e_px, ((e_px - base_fg_pixels) / base_fg_pixels) * 100.0))
    multi_dilation.append((it, d_it, d_px, ((d_px - base_fg_pixels) / base_fg_pixels) * 100.0))

print("\n[Task 8.5] Iteration Experimentation (3x3 Square SE):")
print(f"{'Operation':<10} | {'Iterations':<12} | {'Foreground Px':<15} | {'Area Change (%)':<16} | {'Structural Impact'}")
print("-" * 95)
for it, _, px, delta in multi_erosion:
    impact = "Removes isolated noise; sharpens contours" if it == 1 else (
             "Thin text strokes disappear; bar edges recede" if it == 2 else "Axis labels fully eliminated; narrow bars fragmented")
    print(f"{'Erosion':<10} | {it:<12} | {px:<15,} | {delta:<+16.2f} | {impact}")
for it, _, px, delta in multi_dilation:
    impact = "Bridges broken font strokes; fills internal pinholes" if it == 1 else (
             "Adjacent characters fuse into solid blobs" if it == 2 else "Adjacent data bars merge into single contiguous mask")
    print(f"{'Dilation':<10} | {it:<12} | {px:<15,} | {delta:<+16.2f} | {impact}")
print("-" * 95)

# Figure 3: Multiple Iterations Grid
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes[0, 0].imshow(binary_base, cmap="gray"); axes[0, 0].set_title(f"Original ({base_fg_pixels:,} px)", fontweight="bold"); axes[0, 0].axis("off")
axes[1, 0].imshow(binary_base, cmap="gray"); axes[1, 0].set_title(f"Original ({base_fg_pixels:,} px)", fontweight="bold"); axes[1, 0].axis("off")

for idx, (it, img_e, px, delta) in enumerate(multi_erosion, start=1):
    axes[0, idx].imshow(img_e, cmap="gray")
    axes[0, idx].set_title(f"Erosion (Iter={it})\n{px:,} px ({delta:+.1f}%)", fontweight="bold")
    axes[0, idx].axis("off")

for idx, (it, img_d, px, delta) in enumerate(multi_dilation, start=1):
    axes[1, idx].imshow(img_d, cmap="gray")
    axes[1, idx].set_title(f"Dilation (Iter={it})\n{px:,} px ({delta:+.1f}%)", fontweight="bold")
    axes[1, idx].axis("off")

plt.tight_layout()
fig4_path = os.path.join(output_dir, "task8_iterations_experiment.png")
plt.savefig(fig4_path, dpi=250)
plt.close()
print(f"  -> Saved Iterations Experiment plot to: {fig4_path}")

# -------------------------------------------------------------------------
# Interactive CLI Demo (Task 8 demonstration tool)
# -------------------------------------------------------------------------
def interactive_demo():
    print("\n" + "=" * 55)
    print("TASK 8: INTERACTIVE MORPHOLOGY EXPLORER")
    print("=" * 55)
    op = input("Select Operation (1: Erosion, 2: Dilation, 3: Boundary): ").strip() or "1"
    se_ch = input("Select SE (1: 3x3 Square, 2: 3x3 Cross, 3: 5x5 Square): ").strip() or "1"
    it = int(input("Enter iterations (1-5, default 1): ").strip() or "1")

    se_dict = {"1": se_3x3_rect, "2": se_cross, "3": se_5x5_rect}
    se_used = se_dict.get(se_ch, se_3x3_rect)

    if op == "1":
        res = cv2.erode(binary_base, se_used, iterations=it)
        op_name = f"eroded_se{se_ch}_iter{it}"
    elif op == "2":
        res = cv2.dilate(binary_base, se_used, iterations=it)
        op_name = f"dilated_se{se_ch}_iter{it}"
    else:
        er = cv2.erode(binary_base, se_used, iterations=it)
        res = cv2.subtract(binary_base, er)
        op_name = f"boundary_se{se_ch}_iter{it}"

    out_file = os.path.join(output_dir, f"interactive_{op_name}_{img_name}")
    cv2.imwrite(out_file, res)
    print(f"Result ({np.sum(res==255):,} px) saved successfully to: {out_file}")

if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_demo()
else:
    print("\nBatch analysis complete. Run interactive demo live anytime with:")
    print("python experiments\\task8_morphology\\task8_analysis.py --interactive")
print("=" * 95)
