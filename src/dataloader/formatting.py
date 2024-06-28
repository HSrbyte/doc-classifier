import numpy as np
import tensorflow as tf

from typing import Union, List


def image_normalize(image: np.ndarray,
                    mean: Union[List[float], float],
                    std: Union[List[float], float]) -> np.ndarray:
    """
    Normalize an image using provided mean and standard deviation.

    If a single float value is provided for mean & std, it is applied to all channels.
    If a list of floats is provided, it should contain the mean value for each channel.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.

    Returns:
        numpy.ndarray: Normalized image.
    """
    image = image.astype(np.float32)
    mean = np.float32(np.array(mean))
    std = np.float32(np.array(std))

    if image.ndim == 3:
        if len(mean) != image.ndim or len(std) != image.ndim:
            raise ValueError("Length of mean and std lists should match the number of channels in the image.")
    else:
        if len(mean) != 1 or len(std) != 1:
            raise ValueError("Single mean and std values should be provided for grayscale images.")

    normalized_image = (image - mean) / std
    return normalized_image


def image_denormalize(image: np.ndarray,
                      mean: Union[List[float], float],
                      std: Union[List[float], float]) -> np.ndarray:
    """
    Denormalize an image using provided mean and standard deviation.

    If a single float value is provided for mean & std, it is applied to all channels.
    If a list of floats is provided, it should contain the mean value for each channel.

    Parameters:
        image (numpy.ndarray): Normalized image as a numpy array.
        mean (float): Mean value used for normalization.
        std (float): Standard deviation value used for normalization.

    Returns:
        numpy.ndarray: Denormalized image.
    """
    mean = np.float32(np.array(mean))
    std = np.float32(np.array(std))

    if image.ndim == 3:
        if len(mean) != image.shape[-1] or len(std) != image.shape[-1]:
            raise ValueError("Length of mean and std lists should match the number of channels in the image.")
    else:
        if len(mean) != 1 or len(std) != 1:
            raise ValueError("Single mean and std values should be provided for grayscale images.")

    denormalized_image = image * std + mean
    return denormalized_image


def tf_image_normalize(image: tf.Tensor,
                       mean: Union[List[float], float],
                       std: Union[List[float], float]) -> tf.Tensor:
    """
    Normalize an image using provided mean and standard deviation.

    If a single float value is provided for mean & std, it is applied to all channels.
    If a list of floats is provided, it should contain the mean value for each channel.

    Parameters:
        image (tf.Tensor): Input image as a numpy array.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.

    Returns:
        tf.Tensor: Normalized image.
    """
    image = tf.cast(image, tf.float32)
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    if len(image.shape) == 3:
        if mean.shape[0] != image.shape[-1] or std.shape[0] != image.shape[-1]:
            raise ValueError("Length of mean and std lists should match the number of channels in the image.")
    else:
        if mean.shape[0] != 1 or std.shape[0] != 1:
            raise ValueError("Single mean and std values should be provided for grayscale images.")
    normalized_image = (image - mean) / std
    return normalized_image