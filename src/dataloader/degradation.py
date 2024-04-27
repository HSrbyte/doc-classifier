import cv2
import numpy as np


def jpg_compress(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """
    Compresses an image using JPEG compression.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        quality (int): Quality of compression, between 0 and 100 (higher is better).

    Returns:
        numpy.ndarray: Compressed image.
    """
    _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if len(image.shape) == 3:
        compressed_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    else:
        compressed_image = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
    return compressed_image


def salt_and_pepper_noise(image: np.ndarray,
                          salt_density: float,
                          pepper_density: float
                          ) -> np.ndarray:
    """
    Add salt and pepper noise to an image.

    Args:
        image (np.ndarray): Input image.
        salt_density (float): Density of adding salt noise (range: [0, 1]).
        pepper_density (float): Density of adding pepper noise (range: [0, 1]).

    Returns:
        np.ndarray: Noisy image.
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.random(image.shape) < salt_density
    pepper_mask = np.random.random(image.shape) < pepper_density
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image


def gaussian_noise(image: np.ndarray, mean: float, std_dev: float) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image (np.ndarray): Input image.
        mean (float): Mean of the Gaussian distribution.
        std_dev (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: Noisy image.
    """
    h, w = image.shape[:2]
    shape = (h, w) if len(image.shape) == 2 else (h, w, 3)
    noise = np.random.normal(mean, std_dev, shape)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image