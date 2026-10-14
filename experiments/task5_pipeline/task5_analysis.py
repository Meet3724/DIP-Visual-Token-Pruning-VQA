import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Ensure access to earlier task modules
sys.path.append(os.path.abspath(os.path.join("experiments", "task3_enhancement")))
sys.path.append(os.path.abspath(os.path.join("experiments", "task4_filtering")))
import enhancement as enh
import filtering as filt

output_dir = os.path.join("outputs", "task5")
os.makedirs(output_dir, exist_ok=True)

print("=" * 95)
print("TASK 5: SPATIAL DOMAIN FILTERING (SHARPENING, COMPARISON & INTEGRATED PIPELINE)")
print("=" * 95)

sample_images = ["chart_01.png", "chart_03.png", "chart_05.png"]

# -------------------------------------------------------------------------
# Mathematical Metric Helpers
# -------------------------------------------------------------------------
def calc_metrics(clean, target):
    mse = np.mean((clean.astype(np.float64) - target.astype(np.float64)) ** 2)
    psnr = 100.0 if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    if len(clean.shape) == 3:
        clean_g = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        target_g = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    else:
        clean_g, target_g = clean, target
    ssim_val = ssim(clean_g, target_g, data_range=255)
    return mse, psnr, ssim_val

# -------------------------------------------------------------------------
# Task 5.1: Gradient-Based Sharpening Function
# -------------------------------------------------------------------------
def gradient_sharpening(image, c=0.5):
    """
    Computes first-order spatial derivatives via Sobel operators:
    M(x,y) = |Sobel_x| + |Sobel_y|
    Sharpened output: g(x,y) = f(x,y) + c * M(x,y)
    """
    img_float = image.astype(np.float32)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = img_float

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # Normalize gradient magnitude
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)

    if len(image.shape) == 3:
        grad_3ch = cv2.merge([grad_norm, grad_norm, grad_norm])
        sharpened = img_float + c * grad_3ch
    else:
        sharpened = img_float + c * grad_norm

    return np.clip(sharpened, 0, 255).astype(np.uint8), np.clip(grad_norm, 0, 255).astype(np.uint8)

# -------------------------------------------------------------------------
# Task 5.1 & 5.2: Multi-Image Filtering Benchmark
# -------------------------------------------------------------------------
benchmark_results = []

for idx, img_name in enumerate(sample_images, start=1):
    img_path = os.path.join("dataset", "images", img_name)
    if not os.path.exists(img_path):
        continue

    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prefix = img_name.split('.')[0]

    # Corrupt with mild impulse noise to test restoration vs. amplification
    np.random.seed(42)
    sp_noisy = filt.add_salt_and_pepper_noise(rgb, salt_prob=0.015, pepper_prob=0.015)

    # 1. Smoothing Filters
    mean_img = filt.mean_filter(sp_noisy, ksize=3)
    gauss_img = filt.gaussian_filter(sp_noisy, ksize=3)
    med_img = filt.median_filter(sp_noisy, ksize=3)

    # 2. Sharpening Filters (evaluated directly on original to test edge enhancement)
    lap_sharp = filt.laplacian_sharpening(rgb)
    grad_sharp, grad_mag = gradient_sharpening(rgb, c=0.35)
    hb_sharp = filt.high_boost_filter(rgb, A=1.2, ksize=5)

    # Figure 1: Sharpening Comparison Grid
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(rgb); axes[0].set_title(f"Original ({img_name})", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(lap_sharp); axes[1].set_title("Laplacian (\u2207\u00b2)", fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(grad_sharp); axes[2].set_title("Gradient (Sobel Edge Boost)", fontweight="bold"); axes[2].axis("off")
    axes[3].imshow(hb_sharp); axes[3].set_title("High-Boost (A = 1.2)", fontweight="bold"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"task5_sharpening_{prefix}.png"), dpi=250)
    plt.close()

    # Compute metrics against ground truth
    filters_eval = [
        ("Mean Filter (3x3)", mean_img, "Smoothing (Linear)"),
        ("Gaussian Filter (3x3)", gauss_img, "Smoothing (Weighted)"),
        ("Median Filter (3x3)", med_img, "Smoothing (Non-linear)"),
        ("Laplacian Filter", lap_sharp, "Sharpening (2nd Order)"),
        ("High-Boost (A=1.2)", hb_sharp, "Sharpening (High-Boost)")
    ]

    for fname, fimg, ftype in filters_eval:
        mse, psnr, ssim_v = calc_metrics(rgb, fimg)
        benchmark_results.append({
            "image": img_name,
            "filter": fname,
            "type": ftype,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim_v
        })

    print(f"[{idx}/3] Benchmarked and saved figures for: {img_name}")

# Print Task 5.2 Comparative Table
print("\n" + "-" * 95)
print(f"{'Test Image':<14} | {'Filter Name':<24} | {'MSE':<10} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'Qualitative Effect'}")
print("-" * 95)
for r in benchmark_results:
    if "Mean" in r['filter']:
        qual = "Blurs text labels; smeared impulses"
    elif "Gaussian" in r['filter']:
        qual = "Soft transitions; noise grain intact"
    elif "Median" in r['filter']:
        qual = "Pristine impulse removal; crisp edges"
    elif "Laplacian" in r['filter']:
        qual = "High edge contrast; can amplify noise"
    else:
        qual = "Strong edge boost; stable canvas"
    print(f"{r['image']:<14} | {r['filter']:<24} | {r['mse']:<10.2f} | {r['psnr']:<12.2f} | {r['ssim']:<10.4f} | {qual}")
print("-" * 95)

# -------------------------------------------------------------------------
# Task 5.5: Integrated 3-Stage Pipeline (Acquisition -> Enhancement -> Filtering)
# -------------------------------------------------------------------------
pipeline_src_name = "chart_01.png"
pipeline_bgr = cv2.imread(os.path.join("dataset", "images", pipeline_src_name))
pipeline_rgb = cv2.cvtColor(pipeline_bgr, cv2.COLOR_BGR2RGB)

# Stage 1: Acquired Raw Image (with mild synthetic transmission degradation)
np.random.seed(101)
stage1_acquired = filt.add_salt_and_pepper_noise(pipeline_rgb, salt_prob=0.01, pepper_prob=0.01)

# Stage 2: Spatial Enhancement (Contrast Stretching to normalize dynamic range)
stage2_enhanced = enh.contrast_stretching(stage1_acquired)

# Stage 3: Spatial Filtering (Median Denoising + Subtle High-Boost Sharpening)
denoised = filt.median_filter(stage2_enhanced, ksize=3)
stage3_final = filt.high_boost_filter(denoised, A=1.15, ksize=3)

# Save Individual Pipeline Stages
cv2.imwrite(os.path.join(output_dir, "pipeline_stage1_acquired.png"), cv2.cvtColor(stage1_acquired, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "pipeline_stage2_enhanced.png"), cv2.cvtColor(stage2_enhanced, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "pipeline_stage3_final.png"), cv2.cvtColor(stage3_final, cv2.COLOR_RGB2BGR))

# Generate Integrated Progression Grid
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(pipeline_rgb); axes[0].set_title("Ground Truth (Clean)", fontweight="bold"); axes[0].axis("off")
axes[1].imshow(stage1_acquired); axes[1].set_title("Stage 1: Acquisition (Input)", fontweight="bold"); axes[1].axis("off")
axes[2].imshow(stage2_enhanced); axes[2].set_title("Stage 2: Enhancement (Contrast)", fontweight="bold"); axes[2].axis("off")
axes[3].imshow(stage3_final); axes[3].set_title("Stage 3: Filtering (Final Improved)", fontweight="bold"); axes[3].axis("off")
plt.tight_layout()
pipeline_fig_path = os.path.join(output_dir, "task5_integrated_pipeline.png")
plt.savefig(pipeline_fig_path, dpi=250)
plt.close()
print(f"\n[Task 5.5] Saved 3-Stage Integrated Pipeline figure to: {pipeline_fig_path}")

# -------------------------------------------------------------------------
# Task 5.4: Interactive Filter Selection Interface
# -------------------------------------------------------------------------
def interactive_filter_menu():
    print("\n" + "=" * 55)
    print("TASK 5.4: INTERACTIVE FILTERING & ENHANCEMENT SUITE")
    print("=" * 55)
    img_name = input("Enter image filename from dataset (default: chart_01.png): ").strip() or "chart_01.png"
    filepath = os.path.join("dataset", "images", img_name)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    img = cv2.imread(filepath)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("\nSelect Processing Filter:")
    print("1. Mean Filter (Box Smoothing)")
    print("2. Gaussian Filter")
    print("3. Median Filter (Impulse Denoising)")
    print("4. Laplacian Sharpening")
    print("5. Gradient Sharpening (Sobel)")
    print("6. High-Boost Filtering")
    print("7. Complete 3-Stage Pipeline (Acquire -> Enhance -> Filter)")
    choice = input("Enter choice (1-7): ").strip()

    if choice == "1":
        k = int(input("Kernel size (3 or 5, default 3): ").strip() or "3")
        res = filt.mean_filter(rgb, ksize=k)
        suffix = f"mean_{k}x{k}"
    elif choice == "2":
        k = int(input("Kernel size (3 or 5, default 3): ").strip() or "3")
        res = filt.gaussian_filter(rgb, ksize=k)
        suffix = f"gaussian_{k}x{k}"
    elif choice == "3":
        k = int(input("Kernel size (3 or 5, default 3): ").strip() or "3")
        res = filt.median_filter(rgb, ksize=k)
        suffix = f"median_{k}x{k}"
    elif choice == "4":
        res = filt.laplacian_sharpening(rgb)
        suffix = "laplacian"
    elif choice == "5":
        c = float(input("Gradient weight c (default 0.35): ").strip() or "0.35")
        res, _ = gradient_sharpening(rgb, c=c)
        suffix = f"gradient_c{c}"
    elif choice == "6":
        a = float(input("Amplification factor A (default 1.2): ").strip() or "1.2")
        res = filt.high_boost_filter(rgb, A=a)
        suffix = f"highboost_A{a}"
    elif choice == "7":
        enhanced = enh.contrast_stretching(rgb)
        denoised = filt.median_filter(enhanced, ksize=3)
        res = filt.high_boost_filter(denoised, A=1.15)
        suffix = "full_pipeline"
    else:
        print("Invalid selection.")
        return

    out_path = os.path.join(output_dir, f"interactive_{suffix}_{img_name}")
    cv2.imwrite(out_path, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    print(f"Processed image saved successfully to: {out_path}")

if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_filter_menu()
else:
    print("\nBatch pipeline finished. To run the Task 5.4 interactive menu live, run:")
    print("python experiments\\task5_pipeline\\task5_analysis.py --interactive")

print("=" * 95)
