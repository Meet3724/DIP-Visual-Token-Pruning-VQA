import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths
input_image_path = os.path.join("dataset", "images", "chart_01.png")
output_dir = os.path.join("outputs", "task2")
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("TASK 2: IMAGE ACQUISITION, REPRESENTATION, AND IMAGE FUNDAMENTALS")
print("=" * 80)

# ------------------------------------------------------------------------------
# Task 2.3: Image Acquisition
# ------------------------------------------------------------------------------
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Input image not found at {input_image_path}")

# Read image in BGR format via OpenCV
bgr_img = cv2.imread(input_image_path)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Save original acquisition output
acq_output_path = os.path.join(output_dir, "task2_3_original_acquired.png")
cv2.imwrite(acq_output_path, bgr_img)
print(f"\n[Task 2.3] Image Acquired successfully from: {input_image_path}")
print(f"[Task 2.3] Saved duplicate copy to: {acq_output_path}")

# ------------------------------------------------------------------------------
# Task 2.4: Image Representation
# ------------------------------------------------------------------------------
height, width, channels = bgr_img.shape
total_pixels = height * width
dtype = bgr_img.dtype

# Sample coordinates for pixel inspection (center of image)
test_y, test_x = height // 2, width // 2
bgr_val = bgr_img[test_y, test_x]
rgb_val = rgb_img[test_y, test_x]

print("\n[Task 2.4] Image Representation Metrics:")
print(f"  • Dimensions (Height x Width) : {height} x {width} pixels")
print(f"  • Number of Channels          : {channels}")
print(f"  • Total Resolution (Pixels)   : {total_pixels:,} pixels")
print(f"  • Underlying Data Type        : {dtype} (Range: 0 - 255)")
print(f"  • Pixel Value at ({test_x}, {test_y}) [BGR]: B={bgr_val[0]}, G={bgr_val[1]}, R={bgr_val[2]}")
print(f"  • Pixel Value at ({test_x}, {test_y}) [RGB]: R={rgb_val[0]}, G={rgb_val[1]}, B={rgb_val[2]}")

# ------------------------------------------------------------------------------
# Task 2.5: Color Space Conversion (RGB, Grayscale, HSV)
# ------------------------------------------------------------------------------
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

# Save individual transformed representations
cv2.imwrite(os.path.join(output_dir, "task2_5_grayscale.png"), gray_img)
cv2.imwrite(os.path.join(output_dir, "task2_5_hsv.png"), hsv_img)

# Generate comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb_img)
axes[0].set_title("Original (RGB Color Space)", fontsize=11, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(gray_img, cmap="gray")
axes[1].set_title("Grayscale (Luminance Only)", fontsize=11, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(hsv_img)
axes[2].set_title("HSV Color Space (Hue-Sat-Val)", fontsize=11, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
colorspace_fig_path = os.path.join(output_dir, "task2_5_colorspace_comparison.png")
plt.savefig(colorspace_fig_path, dpi=300)
plt.close()
print(f"\n[Task 2.5] Saved Color Space comparison figure to: {colorspace_fig_path}")

# ------------------------------------------------------------------------------
# Task 2.6: Sampling and Quantization
# ------------------------------------------------------------------------------
# Part A: Spatial Sampling (Resizing: 100%, 50%, 25%)
w_50, h_50 = int(width * 0.50), int(height * 0.50)
w_25, h_25 = int(width * 0.25), int(height * 0.25)

# Downsample then nearest-neighbor scale back up to original size for visual comparison
img_100 = rgb_img
down_50 = cv2.resize(rgb_img, (w_50, h_50), interpolation=cv2.INTER_AREA)
img_50_vis = cv2.resize(down_50, (width, height), interpolation=cv2.INTER_NEAREST)

down_25 = cv2.resize(rgb_img, (w_25, h_25), interpolation=cv2.INTER_AREA)
img_25_vis = cv2.resize(down_25, (width, height), interpolation=cv2.INTER_NEAREST)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_100)
axes[0].set_title(f"100% Sampling ({width}x{height})", fontsize=11, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(img_50_vis)
axes[1].set_title(f"50% Sampling ({w_50}x{h_50})", fontsize=11, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(img_25_vis)
axes[2].set_title(f"25% Sampling ({w_25}x{h_25})", fontsize=11, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
sampling_fig_path = os.path.join(output_dir, "task2_6_sampling_comparison.png")
plt.savefig(sampling_fig_path, dpi=300)
plt.close()

# Part B: Gray-Level Quantization (256, 128, 64, 32 levels)
def quantize_gray(image, levels):
    # Step size: 256 / levels
    step = 256.0 / levels
    # Quantize by dividing, flooring, and scaling back
    quantized = np.floor(image.astype(np.float32) / step) * (255.0 / (levels - 1))
    return np.clip(quantized, 0, 255).astype(np.uint8)

q_256 = quantize_gray(gray_img, 256)
q_128 = quantize_gray(gray_img, 128)
q_64  = quantize_gray(gray_img, 64)
q_32  = quantize_gray(gray_img, 32)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(q_256, cmap="gray")
axes[0].set_title("Quantization: 256 Levels (8-bit)", fontsize=10, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(q_128, cmap="gray")
axes[1].set_title("Quantization: 128 Levels (7-bit)", fontsize=10, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(q_64, cmap="gray")
axes[2].set_title("Quantization: 64 Levels (6-bit)", fontsize=10, fontweight="bold")
axes[2].axis("off")

axes[3].imshow(q_32, cmap="gray")
axes[3].set_title("Quantization: 32 Levels (5-bit)", fontsize=10, fontweight="bold")
axes[3].axis("off")

plt.tight_layout()
quant_fig_path = os.path.join(output_dir, "task2_6_quantization_comparison.png")
plt.savefig(quant_fig_path, dpi=300)
plt.close()
print(f"[Task 2.6] Saved Sampling & Quantization plots to {output_dir}")

# ------------------------------------------------------------------------------
# Task 2.7: Image File Formats & Compression Comparison
# ------------------------------------------------------------------------------
bmp_path = os.path.join(output_dir, "task2_7_format.bmp")
png_path = os.path.join(output_dir, "task2_7_format.png")
jpg_path = os.path.join(output_dir, "task2_7_format.jpg")

# Save in BMP (uncompressed raw format)
cv2.imwrite(bmp_path, bgr_img)

# Save in PNG (lossless compression)
cv2.imwrite(png_path, bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Save in JPEG (lossy compression at standard quality 85)
cv2.imwrite(jpg_path, bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

size_bmp = os.path.getsize(bmp_path)
size_png = os.path.getsize(png_path)
size_jpg = os.path.getsize(jpg_path)

print("\n[Task 2.7] File Format Storage and Compression Benchmark:")
print(f"{'Format':<8} | {'File Size (Bytes)':<18} | {'File Size (KB)':<14} | {'Compression Type':<20}")
print("-" * 70)
print(f"{'BMP':<8} | {size_bmp:<18,} | {size_bmp / 1024:<14.2f} | {'Uncompressed Raster':<20}")
print(f"{'PNG':<8} | {size_png:<18,} | {size_png / 1024:<14.2f} | {'Lossless (DEFLATE)':<20}")
print(f"{'JPEG':<8} | {size_jpg:<18,} | {size_jpg / 1024:<14.2f} | {'Lossy (DCT-Quantized)':<20}")

# Read back JPEG to compute Peak Signal-to-Noise Ratio (PSNR) against original
jpg_decompressed = cv2.imread(jpg_path)
mse = np.mean((bgr_img.astype(np.float64) - jpg_decompressed.astype(np.float64)) ** 2)
psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float("inf")
print(f"\n  • JPEG Artifacts Distortion (PSNR vs. Lossless Original): {psnr:.2f} dB (MSE: {mse:.4f})")

print("\nAll Task 2 processing completed successfully!")
print("=" * 80)
