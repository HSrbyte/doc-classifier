import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from PIL import ImageEnhance
from typing import Optional, Tuple, Union

pil_interp_codes = {
    'nearest': Image.NEAREST,
    'box': Image.BOX,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'hamming': Image.HAMMING,
    'lanczos': Image.LANCZOS
}

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


def image_resize(input_image: np.ndarray,
                 output_size: Tuple[int, int] = (224, 224),
                 interpolation: str = 'lanczos',
                 keep_ratio: bool = True,
                 center_image: bool = True,
                 background_color: Union[Tuple[int, int, int], int]=(0, 0, 0)
                 ) -> np.ndarray:
    """
    Resize an input image to the specified output size with optional settings.

    Parameters:
        input_image (PIL.Image.Image or numpy.ndarray): The input image to be resized.
        output_size (tuple): The desired output size as a tuple (width, height).
        interpolation (str): The interpolation method to be used. Possible values are:
                             'nearest', 'box', 'bilinear', 'bicubic', 'hamming', 'lanczos'.
                             Default is 'lanczos'.
        keep_ratio (bool): Whether to maintain the aspect ratio of the input image while resizing.
                           Default is True.
        center_image (bool): Whether to center the resized image within the output size.
                             Default is True.
        background_color (tuple): The RGB color tuple for the background if the output size
                                  is larger than the input image. Default is (0, 0, 0) (black).

    Returns:
        numpy.ndarray: The resized image as a NumPy array.
    """
    # Convert the np.array image into an Image.Image (PIL image)
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(np.uint8(input_image))

    if keep_ratio is True:
        width, height = input_image.size
        width_ratio = output_size[0] / width
        height_ratio = output_size[1] / height
        resizing_ratio = min(width_ratio, height_ratio)

        # Resize the image while maintaining aspect ratio
        resized_image = input_image.resize((int(input_image.width * resizing_ratio),
                                            int(input_image.height * resizing_ratio)),
                                           pil_interp_codes[interpolation])

        # Define background according to the colorspace of the image and the type of background_color parameter
        if input_image.mode == 'RGB':
            if isinstance(background_color, int):
                background_color = tuple([background_color] * 3)
            background = Image.new('RGB', output_size, background_color)

        elif input_image.mode == 'L':
            if not isinstance(background_color, int):
                background_color = sum(list(background_color))//3
            background = Image.new('L', output_size, background_color)

        # Center the image
        if center_image == True:
            im_x, im_y = resized_image.size
            paste_x = (output_size[0] - im_x) // 2
            paste_y = (output_size[1] - im_y) // 2
            background.paste(resized_image, (paste_x, paste_y))
        else:
            background.paste(resized_image)

        return np.array(background)

    else:
        resized_image = input_image.resize(
            output_size, pil_interp_codes[interpolation])

        return np.array(resized_image)


def image_flip(image: np.ndarray, direction: str) -> np.ndarray:
    """
    Flip an image in the specified direction.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        direction (str): Direction to flip the image.
            Options: 'horizontal', 'vertical', 'diagonal'

    Returns:
        numpy.ndarray: Flipped image.
    Raises:
        ValueError: If an invalid direction is provided.
    """
    if direction == 'horizontal':
        return np.flip(image, axis=1)
    elif direction == 'vertical':
        return np.flip(image, axis=0)
    elif direction == 'diagonal':
        return np.flipud(np.fliplr(image))
    else:
        raise ValueError("Invalid direction. Please choose from 'horizontal', 'vertical', or 'diagonal'.")


def image_merge(foreground: Union[np.ndarray, Image.Image],
                background: Union[np.ndarray, Image.Image],
                mask: Optional[Union[np.ndarray, Image.Image]] = None,
                position: Tuple[int, int] = (0, 0),
                opacity: float = 1
                ) -> np.ndarray:
    """
    Merge a foreground image onto a background image with a specified opacity.

    Parameters:
    - foreground: np.array or PIL.Image.Image
        The image to be merged onto the background.
    - background: np.array or PIL.Image.Image
        The background image onto which the foreground will be merged.
    - mask: PIL.Image.Image, optional
        A mask image to define the transparency of the foreground image.
    - position: tuple, optional
        The position on the background image where the top-left corner of the foreground will be placed.
    - opacity: float, optional
        Opacity of the foreground image. Should be between 0 and 1. Default is 1.

    Returns:
    - np.array
        The merged image as a NumPy array.

    Note:
    - Opacity should be a float between 0 and 1, where 0 is completely transparent and 1 is completely opaque.
    """

    if not isinstance(foreground, Image.Image):
        foreground = Image.fromarray(np.uint8(foreground))
    if not isinstance(background, Image.Image):
        background = Image.fromarray(np.uint8(background))

    if mask is None:
        mask = Image.new("L", foreground.size, round(opacity*255))
    else:
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.uint8(mask))

        if mask.mode != "L":
            mask = mask.convert("L")

        mask = ImageEnhance.Brightness(mask).enhance(opacity)

    background.paste(foreground, position, mask)

    return np.array(background)



def tf_image_flip(image: tf.Tensor, direction: int) -> tf.Tensor:
    """
    Flip an image in the specified direction.

    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        direction (int): Direction to flip the image.
            Options: 0:'horizontal', 1:'vertical', 2:'diagonal'

    Returns:
        numpy.ndarray: Flipped image.
    Raises:
        ValueError: If an invalid direction is provided.
    """
    if direction == 0:
        return tf.image.flip_left_right(image)
    elif direction == 1:
        return tf.image.flip_up_down(image)
    elif direction == 2:
        return tf.image.flip_left_right(tf.image.flip_up_down(image))
    else:
        return image