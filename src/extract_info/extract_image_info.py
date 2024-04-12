import pandas as pd
from src import bluriness
from src import gp_name
import os
import cv2

current_dir = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(current_dir))


def extract_image_info(folder_paths):
    """Extracts information from images in the specified folder paths.

    Parameters:
        folder_paths (list): A list of strings representing the paths to the folders containing images.

    Returns:
        pandas.DataFrame: A DataFrame containing information about the images, including their names, extensions,
        colorspaces, dimensions, bluriness values, relative paths, and dataset names.

    This function iterates through each folder specified in `folder_paths` and then iterates through the files within
    each folder. It checks if each file has a supported image extension and extracts information about the image
    using OpenCV. Information extracted includes the image name, extension, colorspace, height, width, bluriness
    value, relative path with respect to the `project_dir`, and dataset name.

    Supported image extensions: ['.png', '.jpg', '.jpeg', '.tif', '.raw', '.psd', '.svg', '.bmp']

    Note:
        This function assumes that the `project_dir` variable is defined globally.

    Example:
        >>> folder_paths = ['/path/to/folder1', '/path/to/folder2']
        >>> image_info = extract_image_info(folder_paths)
        >>> print(image_info.head())
            image_name extension colorspace  height  width  bluriness                  path dataset
        0  image1.jpg       jpg        rgb    1080   1920   0.123456  /path/to/folder1/image1.jpg  data1
        1  image2.png       png        rgb     720   1280   0.654321  /path/to/folder1/image2.png  data2
    """
    # Create a list to store image information
    image_info_list = []

    # List of supported image extensions
    supported_extensions = ['.png', '.jpg', '.jpeg',
                            '.tif', '.raw', '.psd', '.svg', '.bmp']

    # Iterate through folders containing images
    for folder_path in folder_paths:
        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            # Check if it's an image file
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                # Absolute and relative path of the image
                image_path = os.path.join(folder_path, filename)

                # Load the image with OpenCV
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                if image is not None:
                    # Retrieve image dimensions (condition to check for dimensions)
                    height, width = image.shape[:2]
                    # Number of channels (or 1 if it's a grayscale image)
                    channels = image.shape[2] if len(image.shape) == 3 else 1
                    colorspace = 'rgb' if channels == 3 else (
                        'gray' if channels == 1 else 'binary')
                    extension = os.path.splitext(filename)[-1]
                    bluriness_value = bluriness(image_path)
                    # Call the gp_name function to get the dataset name
                    dataset = gp_name(image_path)

                    # Add information to the list
                    image_info_list.append(
                        [filename, extension, colorspace, height, width, bluriness_value, image_path, dataset])

    # Create a DataFrame from the information list
    image_info_df = pd.DataFrame(image_info_list, columns=[
                                 'image_name', 'extension', 'colorspace', 'height', 'width', 'bluriness', 'path', 'dataset'])

    return image_info_df
