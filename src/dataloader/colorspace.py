from typing import Optional, Union, Tuple, List

import cv2
import numpy as np



def add_alpha_channel(img: np.ndarray,
                      alpha: Optional[np.ndarray] = None,
                      opacity: int = 1) -> np.ndarray:
    """Add an alpha channel to an RGB image or convert a grayscale image to
    RGBA.

    If the input image is already in RGBA format (four color channels), the
    function returns the input image unchanged. If the input image is a
    grayscale image (single color channel), it converts it to an RGBA image by
    adding an alpha channel.

    Args:
        img (np.ndarray): The input image represented as a NumPy array. It can
            be an RGB image with shape (height, width, 3), or a grayscale image
            with shape (height, width).
        alpha (np.ndarray, optional): The alpha channel to add to the input
            image. If not provided, a fully opaque alpha channel will be created
            with the specified opacity value. Defaults to None.
        opacity (int, optional): The opacity value to set for the alpha channel.
            It should be an integer between 0 (fully transparent) and 1 (fully
            opaque). This parameter is used only when `alpha` is not provided.
            Defaults to 1.

    Returns:
        np.ndarray: An RGBA image as a NumPy array with shape (h, w, 4),
            where the fourth channel (index 3) represents the alpha channel.
    """
    if img.shape[2] == 4:
        return img
    elif len(img.shape) == 2:
        img = gray2rgb(img)

    if alpha is None:
        alpha = np.full((img.shape[0], img.shape[1]), int(opacity*255), dtype=np.uint8)
    else:
        if len(alpha.shape) == 3:
            alpha = rgb2gray(alpha.copy())
    rgba_image = np.dstack((img, alpha))
    return rgba_image


def remove_alpha_channel(
    img: np.ndarray,
    return_alpha: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Remove the alpha channel from an RGBA image.

    Args:
        img (np.ndarray): An RGBA image as a NumPy array with shape (h, w, 4).
        return_alpha (bool, optional): If True, also return the alpha channel
            as a separate array. Default is False.

    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray]: If return_alpha is False,
            returns an RGB image as a NumPy array with shape (h, w, 3), where
            the alpha channel has been removed. If return_alpha is True,
            returns a tuple containing the RGB image and the alpha channel as
            separate arrays.
    """
    if img.shape[2] == 4:
        im = np.array(img[:, :, :3], dtype=img.dtype)
        if return_alpha:
            return im, img[:, :, 3]
        else:
            return im
    else:
        return img


def bgr2gray(img: np.ndarray, keepdim: bool = False) -> np.ndarray:
    """Convert a BGR image to grayscale image.

    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.

    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def rgb2gray(img: np.ndarray, keepdim: bool = False) -> np.ndarray:
    """Convert a RGB image to grayscale image.

    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.

    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def gray2bgr(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to BGR image.

    Args:
        img (ndarray): The input image.

    Returns:
        ndarray: The converted BGR image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img


def gray2rgb(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to RGB image.

    Args:
        img (ndarray): The input image.

    Returns:
        ndarray: The converted RGB image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return out_img