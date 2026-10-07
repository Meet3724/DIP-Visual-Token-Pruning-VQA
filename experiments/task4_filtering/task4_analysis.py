import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.metrics import structural_similarity as ssim

# Import our custom filtering module
sys.path.append(os.path.abspath(os.path.join("experiments", "task4_filtering")))
import filtering as filt

output_dir = os.path.join("outputs", "task4")
os.makedirs(output_dir, exist_ok=True)

print("=" * 90)
print("TASK 4: SPATIAL DOMAIN FILTERING (SMOOTHING, SHARPENING & HIGH-BOOST)")
print("=" * 90)

sample_images = ["chart_01.png", "chart_03.png", "chart_05.png"]

def calc_psnr(clean, target):
    mse = np.mean((clean.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0, 0.0
    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return psnr, mse

def calc_ssim(clean, target):
    # Grayscale SSIM
    if len(clean.shape) == 3:
        clean_g = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        target_g = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    else:
        clean_g, target_g = clean, target
    return ssim(clean_g, target_g, data_range=255)

for idx, img_name in enumerate(sample_images, start=1):
    img_path = os.path.join("dataset", "images", img_name)
    if not os.path.exists(img_path):
        continue
    
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prefix = img_name.split('.')[0]

    # 1. Generate Noisy Versions (Task 4.1)
    np.random.seed(42)
    sp_noisy = filt.add_salt_and_pepper_noise(rgb, salt_prob=0.02, pepper_prob=0.02)
    gauss_noisy = filt.add_gaussian_noise(rgb, mean=0.0, var=0.015)

    # Figure 1: Noise Simulation Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(rgb); axes[0].set_title(f"Clean Original ({img_name})", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(sp_noisy); axes[1].set_title("Salt-and-Pepper Noise (Impulse, p=0.04)", fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(gauss_noisy); axes[2].set_title("Additive Gaussian Noise (\u03c3\u00b2=0.015)", fontweight="bold"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"task4_noise_{prefix}.png"), dpi=250)
    plt.close()

    # 2. Benchmark Smoothing Filters on Salt-and-Pepper Noise (Task 4.2)
    mean_3 = filt.mean_filter(sp_noisy, 3)
    mean_5 = filt.mean_filter(sp_noisy, 5)
    gauss_3 = filt.gaussian_filter(sp_noisy, 3)
    gauss_5 = filt.gaussian_filter(sp_noisy, 5)
    med_3 = filt.median_filter(sp_noisy, 3)
    med_5 = filt.median_filter(sp_noisy, 5)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes[0, 0].imshow(sp_noisy); axes[0, 0].set_title("Noisy Input (S&P)", fontweight="bold"); axes[0, 0].axis("off")
    axes[0, 1].imshow(mean_3); axes[0, 1].set_title("Mean 3x3", fontweight="bold"); axes[0, 1].axis("off")
    axes[0, 2].imshow(gauss_3); axes[0, 2].set_title("Gaussian 3x3", fontweight="bold"); axes[0, 2].axis("off")
    axes[0, 3].imshow(med_3); axes[0, 3].set_title("Median 3x3 (Optimal)", fontweight="bold"); axes[0, 3].axis("off")

    axes[1, 0].imshow(rgb); axes[1, 0].set_title("Ground Truth", fontweight="bold"); axes[1, 0].axis("off")
    axes[1, 1].imshow(mean_5); axes[1, 1].set_title("Mean 5x5", fontweight="bold"); axes[1, 1].axis("off")
    axes[1, 2].imshow(gauss_5); axes[1, 2].set_title("Gaussian 5x5", fontweight="bold"); axes[1, 2].axis("off")
    axes[1, 3].imshow(med_5); axes[1, 3].set_title("Median 5x5", fontweight="bold"); axes[1, 3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"task4_smoothing_sp_{prefix}.png"), dpi=250)
    plt.close()

    # 3. Benchmark Sharpening & High-Boost Filters (Syllabus Core)
    lap_sharp = filt.laplacian_sharpening(rgb)
    unsharp = filt.unsharp_masking(rgb, ksize=5, amount=1.2)
    high_boost = filt.high_boost_filter(rgb, A=1.5, ksize=5)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(rgb); axes[0].set_title(f"Original ({img_name})", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(lap_sharp); axes[1].set_title("Laplacian Sharpening (\u2207\u00b2)", fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(unsharp); axes[2].set_title("Unsharp Masking (k=5)", fontweight="bold"); axes[2].axis("off")
    axes[3].imshow(high_boost); axes[3].set_title("High-Boost (A=1.5)", fontweight="bold"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"task4_sharpening_{prefix}.png"), dpi=250)
    plt.close()

    # Print Quantitative Metrics Table for chart_01
    if idx == 1:
        print(f"\nQUANTITATIVE BENCHMARK ON {img_name} (SALT-AND-PEPPER NOISE RESTORATION):")
        print(f"{'Filter Method':<22} | {'Kernel':<8} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'DIP Observation'}")
        print("-" * 88)
        for label, filt_img, k in [
            ("Noisy Baseline", sp_noisy, "-"),
            ("Mean (Average)", mean_3, "3x3"),
            ("Mean (Average)", mean_5, "5x5"),
            ("Gaussian Filter", gauss_3, "3x3"),
            ("Gaussian Filter", gauss_5, "5x5"),
            ("Median Filter", med_3, "3x3"),
            ("Median Filter", med_5, "5x5"),
        ]:
            psnr_v, _ = calc_psnr(rgb, filt_img)
            ssim_v = calc_ssim(rgb, filt_img)
            obs = "Severe impulse grain" if k == "-" else (
                  "Blurs text numbers" if "Mean" in label else (
                  "Retains noise grain" if "Gaussian" in label else "Completely removes impulses"))
            print(f"{label:<22} | {k:<8} | {psnr_v:<12.2f} | {ssim_v:<8.4f} | {obs}")

    print(f"[{idx}/3] Successfully generated all Task 4 visual grids for: {img_name}")

print("\n" + "=" * 90)
print("All Task 4 spatial domain filtering benchmarks completed successfully!")
print(f"Figures saved in: {output_dir}")
print("=" * 90)
