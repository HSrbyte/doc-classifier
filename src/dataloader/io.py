import cv2
import numpy as np


def image_read(file_path: str, color_mode='rgb') -> np.ndarray:
    """
    Lit une image à partir du chemin spécifié et retourne son tableau NumPy.

    Args:
        file_path (str): Chemin du fichier image.
        color_mode (str): Colorimétrie souhaitée ('rgb' ou 'gray'). Par défaut 'rgb'.

    Returns:
        np.ndarray: Tableau NumPy représentant l'image.
    """
    # Lire l'image à partir du chemin spécifié
    image = cv2.imread(file_path)

    # Vérifier si l'image a été lue correctement
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image à partir de {file_path}")

    # Convertir la colorimétrie si nécessaire
    if color_mode == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_mode == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Colorimétrie non valide. Utilisez 'rgb' ou 'gray'.")

    return image


# image_write