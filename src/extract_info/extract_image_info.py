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
    - folder_paths (list): A list of strings representing the paths to the folders containing images.

    Returns:
    - pandas.DataFrame: A DataFrame containing information about the images, including their names, extensions,
      colorspaces, dimensions, bluriness values, relative paths, and dataset names.

    This function iterates through each folder specified in `folder_paths`, then iterates through the files within 
    each folder. It checks if each file has a supported image extension and extracts information about the image 
    using OpenCV. Information extracted includes the image name, extension, colorspace, height, width, bluriness 
    value, relative path with respect to the `project_dir`, and dataset name.

    Supported image extensions: ['.png', '.jpg', '.jpeg', '.tif', 'jpeg', 'raw', 'psd', 'svg', 'bmp']
    
    Note: This function assumes that the `project_dir` variable is defined globally.

    Example:
    >>>folder_paths = ['/path/to/folder1', '/path/to/folder2']
    >>>image_info = extract_image_info(folder_paths)
    >>>print(image_info.head())
        image_name extension colorspace  height  width  bluriness                  path dataset
    0  image1.jpg       jpg        rgb    1080   1920   0.123456  folder1/image1.jpg   data1
    1  image2.png       png        rgb     720   1280   0.654321  folder1/image2.png   data2
    ..."""
    # Création d'une liste pour stocker les informations des images
    image_info_list = []

    # Liste des extensions d'images supportées
    supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', 'jpeg', 'raw', 'psd', 'svg', 'bmp']

    # Parcourir les dossiers contenant les images
    for folder_path in folder_paths:
        # Parcourir les fichiers dans le dossier
        for filename in os.listdir(folder_path):
            # Vérifier s'il s'agit d'un fichier image
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                # Chemin absolu et relatif de l'image
                image_path = os.path.join(folder_path, filename)
                relative_image_path = os.path.relpath(image_path, project_dir)
              
                # Charger l'image avec OpenCV
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                
                if image is not None:
                    # Récupérer les dimensions de l'image (condition à rajouter sur les dimensions)
                    height, width = image.shape[:2]  # Récupérer uniquement la hauteur et la largeur
                    channels = image.shape[2] if len(image.shape) == 3 else 1  # Nombre de canaux (ou 1 s'il s'agit d'une image en niveaux de gris
                    colorspace = 'rgb' if channels == 3 else ('gray' if channels == 1 else 'binary') 
                    extension = filename.split('.')[-1]
                    bluriness_value = bluriness(image_path)
                    dataset = gp_name(image_path)  # Appeler la fonction gp_name pour obtenir le nom du jeu de données
                
                    # Ajouter les informations à la liste
                image_info_list.append([filename, extension, colorspace, height, width, bluriness_value, relative_image_path, dataset])

    # Création d'un DataFrame à partir de la liste des informations
    image_info_df = pd.DataFrame(image_info_list, columns=['image_name', 'extension', 'colorspace', 'height', 'width', 'bluriness', 'path', 'dataset'])

    return image_info_df
