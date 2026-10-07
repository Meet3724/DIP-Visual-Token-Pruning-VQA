import cv2
import numpy as np

# -------------------------------------------------------------------------
# Task 3.2: Intensity Transformations
# -------------------------------------------------------------------------

def image_negative(image):
    """Computes Image Negative: s = 255 - r"""
    return 255 - image

def log_transform(image, c=None):
    """Computes Log Transformation: s = c * log(1 + r)"""
    img_float = image.astype(np.float32)
    if c is None:
        c = 255.0 / np.log(1.0 + np.max(img_float) + 1e-6)
    log_img = c * np.log(1.0 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)

def gamma_transform(image, gamma=1.0, c=1.0):
    """Computes Power-Law (Gamma) Transformation: s = c * (r / 255)^gamma * 255"""
    inv_gamma = gamma
    norm_img = image.astype(np.float32) / 255.0
    gamma_img = c * (norm_img ** inv_gamma) * 255.0
    return np.clip(gamma_img, 0, 255).astype(np.uint8)

def contrast_stretching(image, r_min=None, r_max=None):
    """Piecewise / Min-Max Linear Contrast Stretching"""
    img_float = image.astype(np.float32)
    if r_min is None:
        r_min = np.min(img_float)
    if r_max is None:
        r_max = np.max(img_float)
    
    if r_max == r_min:
        return image
    
    stretched = ((img_float - r_min) / (r_max - r_min)) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)

# -------------------------------------------------------------------------
# Task 3.4: Histogram Equalization
# -------------------------------------------------------------------------

def histogram_equalization(image):
    """
    Applies Global Histogram Equalization.
    Supports both Grayscale (single channel) and RGB (via YUV/Y-channel)
    to prevent unnatural color distortion in charts.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        # Convert to YUV color space, equalize luminance (Y), convert back to BGR
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# -------------------------------------------------------------------------
# Task 3.5: Arithmetic Operations
# -------------------------------------------------------------------------

def image_addition(img1, img2, alpha=0.5, beta=0.5):
    """Blends two images with weighted addition: g(x,y) = alpha*f1 + beta*f2"""
    # Resize img2 to match img1 if dimensions differ
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.addWeighted(img1, alpha, img2, beta, 0)

def image_subtraction(img1, img2):
    """Subtracts img2 from img1: g(x,y) = |f1(x,y) - f2(x,y)|"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.absdiff(img1, img2)

def image_averaging(images):
    """Computes pixel-wise average across a sequence of images to suppress noise"""
    target_shape = (images[0].shape[1], images[0].shape[0])
    resized_images = [cv2.resize(img, target_shape) if img.shape[:2] != images[0].shape[:2] else img for img in images]
    avg = np.mean(resized_images, axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)
