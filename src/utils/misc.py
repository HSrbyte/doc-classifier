import os
import json
import tarfile

from typing import Union, List


def extract_tar_gz(file_path: str, extract_path:str) -> None:
    """
    Extracts a .tar.gz file to the specified path.

    Args:
        file_path (str): Path to the .tar.gz file.
        extract_path (str): Path where the contents will be extracted.

    Returns:
        None
    """
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        print(f"Extraction successful. '{file_path}' extracted to '{extract_path}'")
    except Exception as e:
        print(f"Extraction failed: {e}")


def save_jsonfile(file_path: str, data: Union[List, dict], **kwargs) -> None:
    """Save data to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        data (Union[List, dict]): The data to be saved to the JSON file.
        **kwargs: Additional keyword arguments to pass to json.dump().

    Returns:
        None
    """
    with open(file_path, 'w', encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, **kwargs)


def read_jsonfile(file_path: str, **kwargs) -> Union[List, dict]:
    """Read data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        **kwargs: Additional keyword arguments to pass to json.load().

    Returns:
        Union[List, dict]: The data read from the JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file, **kwargs)
            return data
    except FileNotFoundError:
        raise FileNotFoundError("The specified file is not found.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError("The JSON file is malformed.", "", 0)
