import pandas as pd
import cv2
import matplotlib.pyplot as plt 
import seaborn as sns


def compute_color_histogram(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Convertir l'image de BGR à RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Séparer les canaux de couleurs
    channels = cv2.split(image)
    # Calculer les histogrammes pour chaque canal de couleur
    histograms = [cv2.calcHist([channel], [0], None, [256], [0, 256]) for channel in channels]
    return histograms