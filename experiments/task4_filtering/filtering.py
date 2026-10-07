import cv2
import numpy as np

# -------------------------------------------------------------------------
# Task 4.1: Noise Simulation Models
# -------------------------------------------------------------------------

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Simulates impulse noise by setting random pixels to 255 (salt) or 0 (pepper).
    """
    noisy = image.copy()
    num_salt = np.ceil(salt_prob * image.shape[0] * image.shape[1])
    num_pepper = np.ceil(pepper_prob * image.shape[0] * image.shape[1])

    # Salt (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[tuple(coords)] = 255

    # Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[tuple(coords)] = 0
    return noisy

def add_gaussian_noise(image, mean=0.0, var=0.01):
    """
    Simulates additive Gaussian noise with zero mean and defined variance.
    """
    sigma = var ** 0.5
    img_norm = image.astype(np.float32) / 255.0
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip((img_norm + gauss) * 255.0, 0, 255).astype(np.uint8)
    return noisy

# -------------------------------------------------------------------------
# Task 4.2: Smoothing (Low-Pass) Filters
# -------------------------------------------------------------------------

def mean_filter(image, ksize=3):
    """Spatial Box / Mean Filter (Equal weighting: 1 / k^2)"""
    return cv2.blur(image, (ksize, ksize))

def gaussian_filter(image, ksize=3, sigma=0):
    """Spatial Gaussian Smoothing Filter"""
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)

def median_filter(image, ksize=3):
    """Non-linear Order-Statistic Median Filter"""
    return cv2.medianBlur(image, ksize)

# -------------------------------------------------------------------------
# Core Syllabus: Sharpening and High-Boost Filtering
# -------------------------------------------------------------------------

def laplacian_sharpening(image):
    """
    Sharpens image using second-order Laplacian derivative:
    g(x,y) = f(x,y) - \nabla^2 f(x,y)
    """
    # Use standard 3x3 Laplacian kernel with center negative (-4 or -8)
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def unsharp_masking(image, ksize=5, sigma=1.0, amount=1.0):
    """
    Unsharp Masking:
    f_mask = f(x,y) - \bar{f}(x,y)
    g(x,y) = f(x,y) + amount * f_mask
    """
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    mask = cv2.subtract(image, blurred)
    sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)
    return sharpened

def high_boost_filter(image, A=1.5, ksize=5, sigma=1.0):
    """
    High-Boost Filtering:
    f_hb = A * f(x,y) - \bar{f}(x,y) = (A - 1)*f(x,y) + f_mask(x,y)
    """
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    mask = image.astype(np.float32) - blurred.astype(np.float32)
    boosted = (A * image.astype(np.float32)) - blurred.astype(np.float32)
    return np.clip(boosted, 0, 255).astype(np.uint8)
