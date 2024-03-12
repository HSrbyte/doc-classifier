import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from typing import Tuple, Union, Optional, List


def draw_tesseract_result(
    data: pd.DataFrame,
    image: np.ndarray,
    bbox_color: Tuple[int, int, int] = (255,0,0),
    bbox_thickness: int = 1
    ) -> np.ndarray:
    """Draw bounding boxes and text annotations on the input image based on the
    data provided.

    Parameters:
        data (pd.DataFrame): DataFrame containing bounding box information and
            text annotations.
        image (np.ndarray): Input image on which the bounding boxes and text
            will be drawn.
        bbox_color (Tuple[int, int, int], optional): Color of the bounding
            boxes in BGR format. Defaults to (255, 0, 0) which is blue.
        bbox_thickness (int, optional): Thickness of the bounding boxes' lines.
            Defaults to 1.

    Returns:
        np.ndarray: Image with bounding boxes and text annotations drawn.
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


def plot_image(
    image: Union[np.ndarray, str],
    size: Tuple[int,int],
    title: Optional[str] = None,
    axis: bool = False,
    flip_image_layers: bool = True
    ) -> None:
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


def image_grid_sample(
    images: Union[List[str], List[np.ndarray]],
    height: int,
    width: int,
    square_size: int = 128,
    bg_color: Union[Tuple[int, int, int], int] = 125,
    img_layout: str = "auto",
    grid: bool = True,
    grid_color: Union[Tuple[int, int, int], int] = 0,
    grid_thickness: int = 1,
    seed: int = -1,
    ) -> np.ndarray:
    """Create a sample image by arranging a grid of input images.

    Parameters:
        images (Union[List[str], List[np.ndarray]]): List of input images as
            file paths or numpy arrays.
        height (int): Height of the grid (number of rows).
        width (int): Width of the grid (number of columns).
        square_size (int, optional): Size of each square in the grid. Defaults
            to 128 pixels.
        bg_color (Union[Tuple[int, int, int], int], optional): Background color
            of the output image. Defaults to 125 (gray).
        img_layout (str, optional): Layout method for arranging images in the
            grid. Options are "auto" (automatic layout) or "center" (centered
            layout). Defaults to "auto".
        grid (bool, optional): Flag to indicate whether to draw a grid over the
            output image. Defaults to True.
        grid_color (Union[Tuple[int, int, int], int], optional): Color of the
            grid lines. Defaults to 0 (black).
        grid_thickness (int, optional): Thickness of the grid lines in pixels.
            Defaults to 1.
        seed (int, optional): Random seed for shuffling the input images.
            Defaults to -1 (no seed).

    Returns:
        np.ndarray: A numpy array representing the generated sample image.

    Notes:
        - If img_layout is "auto", images will be placed in the grid
          automatically (on the top-left corner).
        - If img_layout is "center", images will be centered in each square of
          the grid.
        - If the number of input images is greater than the grid size, a random
          subset of images will be selected for the grid based on the provided
          seed.
        - The output image will have a background color specified by bg_color.
        - If grid is True, a grid will be drawn over the output image with the
          specified color and thickness.
    """
    # check img_layout
    if img_layout not in ["auto", "center"]:
        print(f"Not supported method: {img_layout}. Using default method: auto")
        img_layout = "auto"

    # random sample
    if len(images) > height * width:
        if seed == -1:
            seed = random.randint(0, 9999999)
        random.seed(seed)
        images = [images.pop(random.randint(0, len(images)-1)) \
            for _ in range(height * width)]

    # Output image
    if isinstance(bg_color, int):
        bg_color = [bg_color]*3
    dim = (int(height * square_size), int(width * square_size), 3)
    img_out = np.full(dim, bg_color, dtype=np.uint8)

    x_offset = 0
    y_offset = 0
    for n, img in enumerate(images, start=1):
        # image read
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        # check layers
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                img = np.array(img[:, :, :3], dtype=img.dtype)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image resize
        h, w = img.shape[:2]
        r = max(h, w) / square_size
        dim = (int(w / r), int(h / r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        if img_layout == "auto":
            img_out[
                y_offset : y_offset + dim[1],
                x_offset : x_offset + dim[0]
                ] = img
        elif img_layout == "center":
            cy_offset = (square_size - dim[1]) // 2
            cx_offset = (square_size - dim[0]) // 2
            img_out[
                y_offset + cy_offset : y_offset + cy_offset + dim[1],
                x_offset + cx_offset : x_offset + cx_offset + dim[0]
                ] = img

        x_offset += square_size
        if n % width == 0:
            x_offset = 0
            y_offset += square_size

    if not grid:
        return img_out
    # Grid
    if isinstance(grid_color, int):
        grid_color = [grid_color]*3
    for i in range(width+1):
        p1 = [int(i * square_size), 0]
        p2 = [int(i * square_size), int(square_size * height)]
        if i == width:
            p1[0] -= max(1, grid_thickness // 2)
            p2[0] -= max(1, grid_thickness // 2)
        cv2.line(img_out, p1, p2, grid_color, grid_thickness)
    for i in range(height+1):
        p1 = [0, int(i * square_size)]
        p2 = [int(square_size * width), int(i * square_size)]
        if i == height:
            p1[1] -= max(1, grid_thickness // 2)
            p2[1] -= max(1, grid_thickness // 2)
        cv2.line(img_out, p1, p2, grid_color, grid_thickness)

    return img_out