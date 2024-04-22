import os
import cv2
import numpy as np


def image_read(file_path: str, color_mode: str = 'rgb') -> np.ndarray:
    """
    Read an image from the specified path and return its NumPy array.

    Args:
        file_path (str): Path of the image file.
        color_mode (str): Desired color mode ('rgb' or 'gray'). Default is 'rgb'.

    Returns:
        np.ndarray: NumPy array representing the image.
    """
    # Read the image from the specified path
    image = cv2.imread(file_path)

    # Check if the image was read correctly
    if image is None:
        raise FileNotFoundError(f"Unable to read the image from {file_path}")

    # Convert color mode if necessary
    if color_mode == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_mode == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Invalid color mode. Use 'rgb' or 'gray'.")

    return image


def image_save(file_path: str, image: np.ndarray, verbose: int = 0) -> None:
    """
    Saves an image to the specified location.

    Args:
        file_path (str): Full path of the file to save the image.
        image (np.ndarray): NumPy array representing the image.
        verbose (int): Level of verbosity.

    Returns:
        None
    """
    # Extract the directory and file name from the full file path
    directory, file_name = os.path.split(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save the image
    cv2.imwrite(file_path, image)

    if verbose:
        print(f"The image has been successfully saved at '{file_path}'.")