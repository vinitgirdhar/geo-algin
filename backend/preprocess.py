import cv2
import numpy as np
from scipy.ndimage import uniform_filter

def lee_filter(img, window_size=5):
    """
    Apply a Lee Speckle Filter to a single-channel SAR image.
    Works in the linear intensity domain.
    """
    img = img.astype(np.float32)
    img_mean = uniform_filter(img, size=window_size)
    img_sqr_mean = uniform_filter(img**2, size=window_size)
    img_var = img_sqr_mean - img_mean**2

    # Standard speckle coefficient of variation for Sentinel-1 (approx. 0.25)
    noise_std = 0.25
    noise_var = (img_mean * noise_std)**2

    denominator = np.where(img_var == 0, 1e-9, img_var)
    W = (img_var - noise_var) / denominator
    W = np.clip(W, 0, 1)

    img_filtered = img_mean + W * (img - img_mean)
    return img_filtered

def preprocess_s1(img_path, window_size=5, eps=1e-6):
    """
    Preprocess Sentinel-1 SAR image:
    1. Load image as grayscale.
    2. Convert to linear scale in [0, 1].
    3. Apply Lee Filter.
    4. Convert to dB scale (10 * log10(pixels + eps)).
    5. Min-max normalize the dB image to [0, 1].
    6. Expand to 3 channels and apply ImageNet normalization.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Sentinel-1 image not found at {img_path}")
    
    # Linear scale [0, 1]
    img_linear = img.astype(np.float32) / 255.0
    
    # Lee Filter
    img_filtered = lee_filter(img_linear, window_size=window_size)
    
    # Log Scaling (dB scale)
    img_db = 10.0 * np.log10(img_filtered + eps)
    
    # Normalize dB to [0, 1] for CNN input consistency
    min_val = img_db.min()
    max_val = img_db.max()
    denom = max_val - min_val if max_val > min_val else 1.0
    img_db_norm = (img_db - min_val) / denom
    
    # Stack channels to RGB
    img_rgb = np.stack([img_db_norm] * 3, axis=-1)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_rgb - mean) / std
    return img_normalized

def preprocess_s2(img_path):
    """
    Preprocess Sentinel-2 Optical image:
    1. Load image as color (BGR).
    2. Convert BGR to RGB.
    3. Normalize to [0, 1].
    4. Apply standard ImageNet normalization.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Sentinel-2 image not found at {img_path}")
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std
    return img_normalized
