import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike
from typing import Tuple, Union, Optional


def draw_tesseract_result(data: pd.DataFrame,
                          image: ArrayLike,
                          bbox_color: Tuple[int, int, int] = (255,0,0),
                          bbox_thickness: int = 1
                          ) -> ArrayLike:
    """Draw bounding boxes and text annotations on the input image based on the
    data provided.

    Parameters:
        data (pd.DataFrame): DataFrame containing bounding box information and
            text annotations.
        image (ArrayLike): Input image on which the bounding boxes and text
            will be drawn.
        bbox_color (Tuple[int, int, int], optional): Color of the bounding
            boxes in BGR format. Defaults to (255, 0, 0) which is blue.
        bbox_thickness (int, optional): Thickness of the bounding boxes' lines.
            Defaults to 1.

    Returns:
        ArrayLike: Image with bounding boxes and text annotations drawn.
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    title_height = 50
    new_image = np.zeros((h+title_height, w*2, 3), dtype='uint8') + 255

    for row in data.iloc:
        bbox = row[['left', 'top', 'width', 'height']].values
        bbox = [[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]]
        cv2.rectangle(img, bbox[0], bbox[1], bbox_color, bbox_thickness)
        height_text = (((bbox[1][1] - bbox[0][1]) / 25) * (0.8 - 0.20)) + 0.18
        org = (bbox[0][0] + w,
               bbox[0][1] + title_height + int(bbox[1][1] - bbox[0][1]))
        new_image = cv2.putText(new_image, row['text'], org,
                                cv2.FONT_HERSHEY_SIMPLEX, height_text,
                                (0,0,0), 1, cv2.LINE_AA)

    new_image[title_height:h+title_height, :w] = img
    cv2.line(new_image, [w, 0], [w, h+title_height], (0,0,0), 2)
    cv2.rectangle(new_image, [0, 0], [w*2, title_height], (0,0,0), -1)
    new_image = cv2.putText(new_image, "Detection :", (int(w/3), 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255,255,255), 2, cv2.LINE_AA)
    new_image = cv2.putText(new_image, "Recognition :", (int(w/3)+w, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255,255,255), 2, cv2.LINE_AA)
    return new_image


def plot_image(image: Union[np.ndarray, str],
               size: Tuple[int,int],
               title: Optional[str] = None,
               axis: bool = False,
               flip_image_layers: bool = True) -> None:
    """Plot the input image.

    Parameters:
        image (Union[np.ndarray, str]): Input image as a numpy array or file
            path.
        size (Tuple[int,int]): Size of the plot in inches.
        title (Optional[str], optional): Title of the plot. Defaults to None.
        flip_image_layers (bool, optional): Flag to flip image layers for
            proper visualization. Defaults to True.

    Returns:
        None
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    plt.figure(figsize=size)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        if image.shape[2] == 4:
            image = np.array(image[:, :, :3], dtype=image.dtype)
        if flip_image_layers:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis(axis)
    plt.show()