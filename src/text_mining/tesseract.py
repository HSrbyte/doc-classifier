import os
import cv2
import numpy as np
import pandas as pd

import pytesseract
from pytesseract import Output

from tqdm import tqdm
from typing import Union, Tuple, Optional

from src import image_rotate


_columns_names = ["level","page_num","block_num","par_num","line_num",
                 "word_num","left","top","width","height","conf","text"]
empyt_tesseract_ocr_result = pd.DataFrame({name: [] for name in _columns_names})


def tesseract_ocr(
    image: Union[np.ndarray, str],
    lang: str = 'eng',
    config: str = '',
    timeout: int = 0,
    show_errors: bool = True,
    **kwargs
    ) -> pd.DataFrame:
    """Perform Optical Character Recognition (OCR) using Tesseract on the input
    image.

    Parameters:
        image (Union[np.ndarray, str]): Input image as a numpy array or file
            path.
        lang (str, optional): Language code for Tesseract to use. Defaults to
            'fra' (French).
        config (str, optional): Additional configuration options for Tesseract.
            Defaults to ''.
        timeout (int, optional): Timeout value (in seconds) for Tesseract
            operation. Defaults to 0 (no timeout).
        show_errors (bool, optional): Flag to control whether to print errors
            encountered during OCR. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to
            pytesseract.image_to_data.

    Returns:
        pd.DataFrame: DataFrame containing OCR results with columns:
            - level: Level of the detected text.
            - page_num: Page number.
            - block_num: Block number.
            - par_num: Paragraph number.
            - line_num: Line number.
            - word_num: Word number.
            - left: Left coordinate of the bounding box.
            - top: Top coordinate of the bounding box.
            - width: Width of the bounding box.
            - height: Height of the bounding box.
            - conf: Confidence score.
            - text: Detected text.

    Notes:
        If the input image is a file path, it will be read using OpenCV and
        converted to RGB colorspace.
        If Tesseract encounters an error during OCR and show_errors is True,
        the error will be printed.
        If show_errors is False, errors will not be printed, and an empty
        DataFrame will be returned.
    """
    if isinstance(image, str):
        # Read image and convert colorspace BRG to RGB
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        result = pytesseract.image_to_data(image,
                                           lang=lang,
                                           config=config,
                                           output_type='data.frame',
                                           timeout=timeout,
                                           **kwargs)
    except pytesseract.TesseractError as error:
        if show_errors:
            print(f"ERROR [{error.status}] - {error.message}")
        result = empyt_tesseract_ocr_result

    return result


def tesseract_osd(
    image: Union[np.ndarray, str],
    lang: str = 'osd',
    config: str = '',
    timeout: int = 0,
    nice: int = 0,
    show_errors: bool = True,
    ) -> dict:
    """Orientation and script detection of the input image using Tesseract.

    Parameters:
        image (Union[np.ndarray, str]): Input image as a numpy array or file
            path.
        lang (str, optional): Language code for Tesseract to use. Defaults to
            'osd'.
        config (str, optional): Additional configuration options for Tesseract.
            Defaults to ''.
        timeout (int, optional): Timeout value (in seconds) for Tesseract
            operation. Defaults to 0 (no timeout).
        nice (int, optional): Adjust the OCR engine processing nice value.
            Defaults to 0.
        show_errors (bool, optional): Flag to control whether to print errors
            encountered during OCR. Defaults to True.

    Returns:
        dict: A dictionary containing the detected orientation and script
        information with the following keys:
            - page_num: Page number.
            - orientation: Detected orientation in degrees (0, 90, 180, 270).
            - rotate: Recommended rotation direction (0 for no rotation, 1 for
                clockwise, -1 for counterclockwise).
            - orientation_conf: Confidence score for the detected orientation.
            - script: Detected script (e.g., 'Latin', 'Arabic').
            - script_conf: Confidence score for the detected script.

    Notes:
        If the input image is a file path, it will be read using OpenCV and
        converted to RGB colorspace.
        If Tesseract encounters an error during OCR and show_errors is True,
        the error will be printed.
        If show_errors is False, errors will not be printed, and default
        orientation information will be returned.
    """
    if isinstance(image, str):
        # Read image and convert colorspace BRG to RGB
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        result = pytesseract.image_to_osd(image,
                                          lang=lang,
                                          config=config,
                                          nice=nice,
                                          output_type=Output.DICT,
                                          timeout=timeout)
    except pytesseract.TesseractError as error:
        if show_errors:
            print(f"ERROR [{error.status}] - {error.message}")
        result = {
            'page_num': 0,
            'orientation': 0,
            'rotate': 0,
            'orientation_conf': 0,
            'script': None,
            'script_conf': 0
        }

    return result


def tesseract_ocr_postprocess(
    data: pd.DataFrame,
    conf: float = 0.5,
    drop_unique_chars: bool = True,
    dropna: bool = True,
    drop_spacebars: bool = True,
    ) -> pd.DataFrame:
    """Postprocess tesseract result.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        conf (float, optional): Confidence threshold for text data,
            defaults to 0.5.
        drop_unique_chars (bool, optional): Flag to drop rows containing only
            unique characters (e.g., punctuation), defaults to True.
        dropna (bool, optional): Flag to drop rows with missing text values,
            defaults to True.
        drop_spacebars (bool, optional): Flag to drop rows containing only
            space characters, defaults to True.

    Returns:
        pd.DataFrame: Processed DataFrame after applying the specified
            transformations.
    """
    if dropna:
        data.dropna(subset='text', inplace=True, ignore_index=True)
    if drop_spacebars:
        data = data[data['text'] != ' ']

    data = data[data['conf'] >= conf*100]

    if drop_unique_chars and not data.empty:
        punctuation_pattern = r'^[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+$©»‘æë°“”€'
        data['text'] = data['text'].astype(str)
        data = data[~data['text'].str.match(punctuation_pattern)]

    data.reset_index(drop=True, inplace=True)

    return data


def tesseract_image_process(
    image: Union[str, np.ndarray],
    confiance: float = 0.5,
    lang_ocr: str = "eng+fra",
    lang_osd: str = "osd",
    config_ocr: str = "--psm 11",
    config_osd: str = "",
    timeout_ocr: int = 60,
    timeout_osd: int = 60,
    show_errors: bool = False
    ) -> Tuple[pd.DataFrame, dict]:
    """Perform image processing and OCR using Tesseract.

    Parameters:
        image (Union[str, np.ndarray]): Input image as a file path or a numpy
            array.
        confiance (float, optional): Confidence threshold for post-processing
            OCR results. Defaults to 0.5.
        lang_ocr (str, optional): Languages for OCR. Defaults to "eng+fra"
            (English and French).
        lang_osd (str, optional): Language for orientation and script detection
            (OSD). Defaults to "osd".
        config_ocr (str, optional): Additional configuration options for OCR.
            Defaults to "--psm 11".
        config_osd (str, optional): Additional configuration options for OSD.
            Defaults to "".
        timeout_ocr (int, optional): Timeout value (in seconds) for OCR
            operation. Defaults to 60 seconds.
        timeout_osd (int, optional): Timeout value (in seconds) for OSD
            operation. Defaults to 60 seconds.
        show_errors (bool, optional): Flag to control whether to print errors
            encountered during processing. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing:
            - OCR result: DataFrame containing OCR results.
            - OSD result: Dictionary containing OSD results.

    Notes:
        - If the input image is a file path, it will be read using OpenCV and
          converted to RGB colorspace.
        - If Tesseract encounters an error during processing and show_errors is
          True, the error will be printed. If show_errors is False, errors will
          not be printed, and default results will be returned.
        - If the orientation detection suggests a non-zero rotation angle,
          another OCR attempt will be made after rotating the image. If the
          rotated OCR result has more detected elements than the original OCR
          result, the rotated result will be used.
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ocr_result = tesseract_ocr(image,
                               lang_ocr,
                               config_ocr,
                               timeout_ocr,
                               show_errors)
    ocr_result = tesseract_ocr_postprocess(ocr_result, confiance)

    osd_result = tesseract_osd(image,
                                lang_osd,
                                config_osd,
                                timeout_osd,
                                show_errors=show_errors)

    if osd_result['rotate'] != 0:
        image_rot = image_rotate(image, osd_result['rotate'], auto_bound=True)
        ocr_result_rot = tesseract_ocr(image_rot,
                                   lang_ocr,
                                   config_ocr,
                                   timeout_ocr,
                                   show_errors)
        ocr_result_rot = tesseract_ocr_postprocess(ocr_result_rot, confiance)
        if ocr_result_rot.size > ocr_result.size:
            ocr_result = ocr_result_rot

    return ocr_result, osd_result


def tesseract_data_process(
    data_path: str,
    image_folder: str = 'images',
    output_path: Optional[str] = None,
    confiance: float = 0.5,
    lang_ocr: str = "eng+fra",
    lang_osd: str = "osd",
    config_ocr: str = "--psm 11",
    config_osd: str = "",
    timeout_ocr: int = 60,
    timeout_osd: int = 60,
    show_errors: bool = False
    ) -> None:
    """Process a dataset of images using Tesseract.

    Parameters:
        data_path (str): Path to the dataset directory containing images.
        output_path (Optional[str], optional): Path to the output directory.
            If None, output will be stored in the same directory as the input
            data. Defaults to None.
        confiance (float, optional): Confidence threshold for post-processing
            OCR results. Defaults to 0.5.
        lang_ocr (str, optional): Languages for OCR. Defaults to "eng+fra"
            (English and French).
        lang_osd (str, optional): Language for orientation and script detection
            (OSD). Defaults to "osd".
        config_ocr (str, optional): Additional configuration options for OCR.
            Defaults to "--psm 11".
        config_osd (str, optional): Additional configuration options for OSD.
            Defaults to "".
        timeout_ocr (int, optional): Timeout value (in seconds) for OCR
            operation. Defaults to 60 seconds.
        timeout_osd (int, optional): Timeout value (in seconds) for OSD
            operation. Defaults to 60 seconds.
        show_errors (bool, optional): Flag to control whether to print errors
            encountered during processing. Defaults to False.

    Notes:
        The function processes each image in the dataset directory using
        tesseract_image_process function. If the output_path is None, the
        output will be stored in the same directory as the input data.
        The processed OCR results are saved as individual CSV files in
        a subdirectory named 'tesseract_ocr'. The OSD results are saved as
        a single CSV file named 'tesseract_ost.csv' in the output_path
        directory.
    """
    if output_path is None:
        output_path = data_path

    out_folder = os.path.join(output_path, 'tesseract_ocr')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    images_dir = os.path.join(data_path, image_folder)
    all_images = os.listdir(images_dir)
    all_images.sort()
    osd_data = []
    for image_name in tqdm(all_images, total=len(all_images)):
        image_path = os.path.join(images_dir, image_name)
        ocr_result, osd_result = tesseract_image_process(
                                    image_path,
                                    confiance,
                                    lang_ocr,
                                    lang_osd,
                                    config_ocr,
                                    config_osd,
                                    timeout_ocr,
                                    timeout_osd,
                                    show_errors)
        osd_result['image_name'] = image_name
        osd_data.append(osd_result)

        name, ext = os.path.splitext(image_name)
        csv_path = os.path.join(out_folder, name + '.csv')
        ocr_result.to_csv(csv_path, index=False)

    df_ost = pd.DataFrame(osd_data)
    df_ost.to_csv(os.path.join(output_path, 'tesseract_ost.csv'), index=False)