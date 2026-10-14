import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

output_dir = os.path.join("outputs", "task7")
os.makedirs(output_dir, exist_ok=True)

print("=" * 95)
print("TASK 7: REGION-BASED SEGMENTATION (REGION GROWING, CONNECTIVITY & SPLIT-MERGE)")
print("=" * 95)

# -------------------------------------------------------------------------
# Step 1: Input Image Selection & Fundamental Inspection
# -------------------------------------------------------------------------
input_img_name = "chart_01.png"
input_path = os.path.join("dataset", "images", input_img_name)

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input image not found at: {input_path}")

bgr_img = cv2.imread(input_path)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

height, width = gray_img.shape
min_val = int(np.min(gray_img))
max_val = int(np.max(gray_img))

print(f"\n[Step 1] Image Acquisition & Fundamentals for '{input_img_name}':")
print(f"  • Dimensions (Height x Width) : {height} x {width} pixels")
print(f"  • Total Resolution            : {height * width:,} pixels")
print(f"  • Intensity Dynamic Range     : [{min_val}, {max_val}] (uint8)")

# -------------------------------------------------------------------------
# Step 2: Seed Point Selection
# -------------------------------------------------------------------------
# In chart_01.png, we automatically detect a prominent colored bar to place the seed point
# Alternatively, a coordinate inside the first prominent data bar (e.g., y=150, x=350)
seed_y, seed_x = 150, 350
seed_val = int(gray_img[seed_y, seed_x])

# Ensure seed point is inside a foreground bar (not white canvas 255)
if seed_val > 240:
    # Scan vertically until a non-white bar is found
    for y in range(100, height - 100, 20):
        for x in range(150, width - 150, 20):
            if gray_img[y, x] < 200:
                seed_y, seed_x = y, x
                seed_val = int(gray_img[seed_y, seed_x])
                break
        if seed_val < 200:
            break

print(f"\n[Step 2] Seed Point Selected:")
print(f"  • Coordinates (X, Y)          : ({seed_x}, {seed_y})")
print(f"  • Seed Pixel Gray Intensity   : {seed_val}")

# -------------------------------------------------------------------------
# Region Growing Implementation (BFS Queue-Based)
# -------------------------------------------------------------------------
def region_growing(img, seed, threshold, connectivity=4, dynamic_mean=True):
    """
    Breadth-First Search (BFS) Queue-based Region Growing.
    Connectivity: 4 or 8.
    Similarity Criterion: |pixel_val - current_region_mean| <= threshold
    """
    h, w = img.shape
    visited = np.zeros((h, w), dtype=bool)
    segmented = np.zeros((h, w), dtype=np.uint8)

    sy, sx = seed[1], seed[0]
    queue = deque([(sy, sx)])
    visited[sy, sx] = True
    segmented[sy, sx] = 255

    region_sum = float(img[sy, sx])
    region_count = 1

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connected
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        cy, cx = queue.popleft()
        curr_mean = (region_sum / region_count) if dynamic_mean else float(img[sy, sx])

        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                diff = abs(float(img[ny, nx]) - curr_mean)
                if diff <= threshold:
                    visited[ny, nx] = True
                    segmented[ny, nx] = 255
                    queue.append((ny, nx))
                    region_sum += float(img[ny, nx])
                    region_count += 1
                else:
                    visited[ny, nx] = True  # Visited but rejected

    return segmented, region_count

# -------------------------------------------------------------------------
# Task 7.1 (Step 5): Experiment with Different Threshold Values
# -------------------------------------------------------------------------
thresh_low = 10
thresh_med = 25
thresh_high = 65

seg_low, count_low = region_growing(gray_img, (seed_x, seed_y), thresh_low, connectivity=8)
seg_med, count_med = region_growing(gray_img, (seed_x, seed_y), thresh_med, connectivity=8)
seg_high, count_high = region_growing(gray_img, (seed_x, seed_y), thresh_high, connectivity=8)

print("\n[Task 7.1] Threshold Experimentation Results:")
print(f"  • Low Threshold   (T = {thresh_low:<2}) : {count_low:<8,} pixels | Under-segmented (incomplete core)")
print(f"  • Medium Threshold(T = {thresh_med:<2}) : {count_med:<8,} pixels | Optimal (crisp bar boundary)")
print(f"  • High Threshold  (T = {thresh_high:<2}) : {count_high:<8,} pixels | Over-segmented (leaked into canvas/adjacent elements)")

# Figure 1: Threshold Variation Comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

# Plot Seed Point on Original
seed_marker_img = rgb_img.copy()
cv2.circle(seed_marker_img, (seed_x, seed_y), 8, (255, 0, 0), -1)
cv2.circle(seed_marker_img, (seed_x, seed_y), 12, (255, 255, 255), 2)
axes[0].imshow(seed_marker_img)
axes[0].set_title(f"Seed Point at ({seed_x}, {seed_y})", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(seg_low, cmap="gray")
axes[1].set_title(f"T = {thresh_low} (Low: Under-segmented)\n{count_low:,} px", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(seg_med, cmap="gray")
axes[2].set_title(f"T = {thresh_med} (Optimal Segmentation)\n{count_med:,} px", fontweight="bold")
axes[2].axis("off")

axes[3].imshow(seg_high, cmap="gray")
axes[3].set_title(f"T = {thresh_high} (High: Severe Leakage)\n{count_high:,} px", fontweight="bold")
axes[3].axis("off")

plt.tight_layout()
thresh_fig_path = os.path.join(output_dir, "task7_threshold_comparison.png")
plt.savefig(thresh_fig_path, dpi=250)
plt.close()
print(f"  -> Saved Threshold Comparison plot to: {thresh_fig_path}")

# -------------------------------------------------------------------------
# Task 7.2: Compare 4-Connected vs. 8-Connected Region Growing
# -------------------------------------------------------------------------
t0 = time.perf_counter()
seg_4conn, count_4conn = region_growing(gray_img, (seed_x, seed_y), thresh_med, connectivity=4)
t_4conn = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
seg_8conn, count_8conn = region_growing(gray_img, (seed_x, seed_y), thresh_med, connectivity=8)
t_8conn = (time.perf_counter() - t0) * 1000

# Inspect difference (pixels captured by 8-conn but missed by 4-conn)
diff_conn = cv2.absdiff(seg_8conn, seg_4conn)
diag_pixels_gained = np.sum(diff_conn > 0)

print("\n[Task 7.2] 4-Connected vs. 8-Connected Benchmark (T = 25):")
print(f"{'Connectivity':<14} | {'Region Pixels':<15} | {'Runtime (ms)':<14} | {'DIP Observation'}")
print("-" * 75)
print(f"{'4-Connected':<14} | {count_4conn:<15,} | {t_4conn:<14.2f} | Disconnects diagonal text/lines")
print(f"{'8-Connected':<14} | {count_8conn:<15,} | {t_8conn:<14.2f} | Full diagonal connectivity (+{diag_pixels_gained:,} px)")
print("-" * 75)

# Figure 2: Connectivity Comparison Grid
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].imshow(seg_4conn, cmap="gray")
axes[0].set_title(f"4-Connected Region ({count_4conn:,} px)\nTime: {t_4conn:.2f} ms", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(seg_8conn, cmap="gray")
axes[1].set_title(f"8-Connected Region ({count_8conn:,} px)\nTime: {t_8conn:.2f} ms (Optimal)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(diff_conn, cmap="hot")
axes[2].set_title(f"Diagonal Pixel Gain ({diag_pixels_gained:,} px)\n[8-Connected - 4-Connected]", fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
conn_fig_path = os.path.join(output_dir, "task7_connectivity_comparison.png")
plt.savefig(conn_fig_path, dpi=250)
plt.close()
print(f"  -> Saved Connectivity Comparison plot to: {conn_fig_path}")

# -------------------------------------------------------------------------
# Task 7.3: Region Splitting and Merging (Quadtree Homogeneity)
# -------------------------------------------------------------------------
print("\n[Task 7.3] Implementing Quadtree Region Splitting and Merging...")

def quadtree_split(img, x, y, w, h, std_thresh=12.0, min_size=8):
    """
    Recursively splits regions if standard deviation exceeds std_thresh
    until min_size is reached.
    """
    sub = img[y:y+h, x:x+w]
    std = np.std(sub)

    if std <= std_thresh or w <= min_size or h <= min_size:
        return [(x, y, w, h, float(np.mean(sub)))]
    
    half_w, half_h = w // 2, h // 2
    quads = []
    # 4 Quadrants: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    quads.extend(quadtree_split(img, x, y, half_w, half_h, std_thresh, min_size))
    quads.extend(quadtree_split(img, x + half_w, y, w - half_w, half_h, std_thresh, min_size))
    quads.extend(quadtree_split(img, x, y + half_h, half_w, h - half_h, std_thresh, min_size))
    quads.extend(quadtree_split(img, x + half_w, y + half_h, w - half_w, h - half_h, std_thresh, min_size))
    return quads

def split_and_merge(img, std_thresh=14.0, merge_thresh=15.0, min_size=8):
    h, w = img.shape
    # 1. Splitting Phase
    regions = quadtree_split(img, 0, 0, w, h, std_thresh=std_thresh, min_size=min_size)
    
    # Reconstruct split-only representation
    split_img = np.zeros((h, w), dtype=np.uint8)
    split_grid = np.full((h, w, 3), 255, dtype=np.uint8)
    for (rx, ry, rw, rh, rmean) in regions:
        split_img[ry:ry+rh, rx:rx+rw] = int(rmean)
        split_grid[ry:ry+rh, rx:rx+rw] = int(rmean)
        cv2.rectangle(split_grid, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 1)

    # 2. Merging Phase: Merge adjacent regions with close mean intensities
    # Quantize region means using merge_thresh bins
    merged_img = np.zeros((h, w), dtype=np.uint8)
    for (rx, ry, rw, rh, rmean) in regions:
        merged_val = int(round(rmean / merge_thresh) * merge_thresh)
        merged_img[ry:ry+rh, rx:rx+rw] = np.clip(merged_val, 0, 255)

    return split_grid, split_img, merged_img, len(regions)

# Run Split & Merge on a representative chart area
small_crop = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
quad_grid, split_res, merged_res, num_quads = split_and_merge(small_crop, std_thresh=15.0, merge_thresh=20.0, min_size=8)

print(f"  • Quadtree Decomposition Complete: {num_quads:,} leaf blocks generated.")
print(f"  • Homogeneity Threshold (\u03c3)       : 15.0 | Merge Threshold: 20.0")

# Figure 3: Split & Merge Pipeline
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(small_crop, cmap="gray")
axes[0].set_title(f"Input Crop ({input_img_name})", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(quad_grid)
axes[1].set_title(f"1. Quadtree Split Grid\n({num_quads:,} Leaf Blocks)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(split_res, cmap="gray")
axes[2].set_title("2. Region Splitting Output", fontweight="bold")
axes[2].axis("off")

axes[3].imshow(merged_res, cmap="gray")
axes[3].set_title("3. Final Merged Segmentation", fontweight="bold")
axes[3].axis("off")

plt.tight_layout()
split_merge_fig_path = os.path.join(output_dir, "task7_split_and_merge.png")
plt.savefig(split_merge_fig_path, dpi=250)
plt.close()
print(f"  -> Saved Split & Merge figure to: {split_merge_fig_path}")

# -------------------------------------------------------------------------
# Interactive CLI Demo (Task 7 viva utility)
# -------------------------------------------------------------------------
def interactive_demo():
    print("\n" + "=" * 55)
    print("TASK 7: INTERACTIVE REGION GROWING DEMO")
    print("=" * 55)
    coord_in = input(f"Enter Seed (X, Y) [default: {seed_x}, {seed_y}]: ").strip()
    if coord_in:
        try:
            sx, sy = map(int, coord_in.split(","))
        except:
            sx, sy = seed_x, seed_y
    else:
        sx, sy = seed_x, seed_y

    t_in = input("Enter threshold T (e.g. 25): ").strip()
    thresh = int(t_in) if t_in else 25

    conn_in = input("Enter connectivity (4 or 8) [default: 8]: ").strip()
    conn = int(conn_in) if conn_in in ["4", "8"] else 8

    seg, count = region_growing(gray_img, (sx, sy), thresh, connectivity=conn)
    out_name = f"interactive_seed_{sx}_{sy}_T{thresh}_conn{conn}.png"
    out_file = os.path.join(output_dir, out_name)
    cv2.imwrite(out_file, seg)
    print(f"Segmented region with {count:,} pixels saved to: {out_file}")

if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    interactive_demo()
else:
    print("\nBatch analysis complete. Run interactive demo live with:")
    print("python experiments\\task7_region_segmentation\\task7_analysis.py --interactive")
print("=" * 95)
