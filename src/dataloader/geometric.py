import cv2
import numpy as np

from typing import Optional, Tuple, Union


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

cv2_border_modes = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
    'reflect_101': cv2.BORDER_REFLECT_101,
    'transparent': cv2.BORDER_TRANSPARENT,
    'isolated': cv2.BORDER_ISOLATED
}

def image_rotate(
    img: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    border_value: int = 0,
    interpolation: str = 'bilinear',
    auto_bound: bool = False,
    border_mode: str = 'constant',
    return_matrix: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Rotate the input image by the specified angle.

    Parameters:
        img (np.ndarray): Input image as a numpy array.
        angle (float): Angle of rotation in degrees (positive for clockwise,
            negative for counterclockwise).
        center (Optional[Tuple[float, float]], optional): Center of rotation
            as (x, y) coordinates. If None, the center is set to the center of
            the image. Defaults to None.
        scale (float, optional): Scale factor for resizing the image after
            rotation. Defaults to 1.0.
        border_value (int, optional): Value used for out-of-bound pixels.
            Defaults to 0.
        interpolation (str, optional): Interpolation method to use for
            resizing. Options: 'nearest', 'bilinear', 'bicubic', 'lanczos'.
            Defaults to 'bilinear'.
        auto_bound (bool, optional): Flag to automatically adjust the output
            image size to fit the rotated image. Defaults to False.
        border_mode (str, optional): Border mode for handling out-of-bound
            pixels. Options: 'constant', 'edge', 'reflect', 'wrap'. Defaults
            to 'constant'.
        return_matrix (bool, optional): Flag to return the transformation
            matrix along with the rotated image. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Rotated image or a
            tuple containing the rotated image and the transformation matrix,
            depending on the value of return_matrix.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    rotated_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(rotated_matrix[0, 0])
        sin = np.abs(rotated_matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        rotated_matrix[0, 2] += (new_w - w) * 0.5
        rotated_matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        rotated_matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderMode=cv2_border_modes[border_mode],
        borderValue=border_value)

    if return_matrix:
        return rotated, rotated_matrix
    else:
        return rotated


def image_flip(image: np.ndarray, direction: str) -> np.ndarray:
    """
    Flip an image in the specified direction.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        direction (str): Direction to flip the image.
            Options: 'horizontal', 'vertical', 'diagonal'

    Returns:
        numpy.ndarray: Flipped image.
    """
    if direction == 'horizontal':
        return np.flip(image, axis=1)
    elif direction == 'vertical':
        return np.flip(image, axis=0)
    elif direction == 'diagonal':
        return np.flipud(np.fliplr(image))
    else:
        raise ValueError("Invalid direction. Please choose from 'horizontal', 'vertical', or 'diagonal'.")
