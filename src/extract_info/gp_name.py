import os 

def gp_name(file_path):
    """
    Obtient le nom du dossier grand-parent du chemin du fichier.

    Args:
        file_path (str): Le chemin du fichier.

    Returns:
        str: Le nom du dossier grand-parent.
    """
    # Vérifier si le chemin du fichier est valide
    if not os.path.exists(file_path):
        raise ValueError("Le chemin du fichier fourni n'existe pas.")
    
    # Obtenir le dossier parent du fichier
    parent_folder = os.path.dirname(file_path)
    
    # Obtenir le dossier parent du dossier parent
    grandparent_folder = os.path.dirname(parent_folder)
    
    # Extraire le nom du dossier grand-parent
    grandparent_directory_name = os.path.basename(grandparent_folder)
    
    return grandparent_directory_name