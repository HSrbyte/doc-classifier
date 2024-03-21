import os
import cv2
from src import bluriness
from src import gp_name

# Ajoutez la fonction manquante gp_name

# Fonction pour l'extraction des informations d'une image
def img_info(image_path):
    # Création d'une liste pour stocker les informations des images
    image_info_list = []

    # Liste des extensions d'images supportées
    supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp', '.raw', '.svg', '.psd']

    # Vérifier s'il s'agit d'un fichier image
    if any(image_path.lower().endswith(ext) for ext in supported_extensions):
        # Charger l'image avec OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is not None:
            # Récupérer les dimensions de l'image
            height, width, channels = image.shape
            colorspace = 'rgb' if channels == 3 else ('gray' if channels == 1 else 'binary') 
            extension = image_path.split('.')[-1]
            bluriness_value = bluriness(image_path)  # Utiliser la fonction bluriness pour calculer le flou
            filename = os.path.basename(image_path)
            dataset = gp_name(image_path)  # Appeler la fonction gp_name pour obtenir le nom du jeu de données
                    
            # Ajouter les informations à la liste
            image_info_list.append([filename, dataset, extension, height, width, colorspace, bluriness_value])
    
    return image_info_list