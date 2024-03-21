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


def get_all_files_from_data_folder(
    data_name: str,
    folder: str,
    data_type: str = "raw",
    only_names: bool = False
    ) -> List[str]:
    """Retrieve the paths or names of all files in a specific folder within a
    data directory.

    Args:
        data_name (str): The name of the dataset.
        folder (str): The name of the folder containing the files.
        data_type (str, optional): The type of data directory (e.g., "raw",
            "processed"). Default is "raw".
        only_names (bool, optional): Whether to return only the names of the
            files. Default is False.

    Returns:
        List[str]: A list of file paths if only_names is False, or a list of
            file names otherwise.
    """
    data_path = os.path.join("data", data_type, data_name, folder)
    all_filesnames = os.listdir(data_path)
    if only_names:
        return all_filesnames
    all_files_paths = [os.path.join(data_path, filename) \
        for filename in all_filesnames]
    return all_files_paths