import os
import cv2

def bluriness(img_path):
    """
    Calcule la netteté de l'image en utilisant la variance du Laplacien.

    Args:
        img_path (str): Le chemin de l'image.

    Returns:
        float: La variance du Laplacien, qui indique la netteté de l'image.
    """
    # Vérifier si le chemin de l'image est valide
    if not os.path.exists(img_path):
        raise ValueError("Le chemin de l'image fourni n'existe pas.")
    
    # Lire l'image en niveaux de gris
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Vérifier si l'image est valide
    if img is None:
        raise ValueError("Impossible de lire l'image. Vérifiez le chemin de l'image.")
    
    # Calculer le Laplacien
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Calculer la variance du Laplacien
    bluriness_var = laplacian.var()
    
    return bluriness_var