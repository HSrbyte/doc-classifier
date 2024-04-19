import cv2
import numpy as np


def image_jpg_compress(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """
    Compresses an image using JPEG compression.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        quality (int): Quality of compression, between 0 and 100 (higher is better).

    Returns:
        numpy.ndarray: Compressed image.
    """
    image.shape
    _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if len(image.shape) == 3:
        compressed_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    else:
        compressed_image = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
    return compressed_image