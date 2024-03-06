import os
import tarfile


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