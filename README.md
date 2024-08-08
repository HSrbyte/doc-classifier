# doc-classifier
Projet de classification de documents dans le cadre de la formation DataScientest (novembre 2023). Le projet est divisé en plusieurs étapes, chacune documentée dans plusieurs notebooks :

### 1 - Analyse exploratoire des datasets (notebooks 00 à 05):
Cette étape inclut la compréhension et l'analyse initiale des jeux de données, avec des visualisations et des statistiques descriptives pour préparer les étapes suivantes.

### 2 - Première modélisation : Classification des données textuelles (notebooks 06 à 11):
Utilisation de Tesseract pour l'extraction de texte à partir des documents, suivie d'une classification basée sur les données textuelles. Les modèles développés dans cette phase sont principalement axés sur le traitement du langage naturel (NLP).

### 3 - Deuxième modélisation : Classification des images (notebooks 12 à 19):
Application de techniques de Deep Learning pour la classification des documents basée sur les images. Cette étape comprend le prétraitement des images, la conception des modèles de réseaux de neurones convolutifs (CNN), et l'évaluation des performances des modèles.


### Organisation des fichiers

- [notebooks/](notebooks/)
Ce dossier contient tous les notebooks détaillant les différentes étapes du projet, du nettoyage des données à la modélisation finale.

- [reports/](reports/)
Le rapport final du projet se trouve dans ce dossier. Il contient une synthèse de toutes nos analyses et conclusions.

- [models/](models/)
Ce dossier doit contenir tous les modèles pré-entraînés. Vous pouvez télécharger tous les modèles pré-entraînés utilisés dans ce projet via le lien suivant : [Google Drive](https://drive.google.com/drive/folders/1Nni1RCqoR4cPTvxcwLCgCB0zghz5uBhi?usp=sharing)



## 🛠️ Installation

Télécharger le projet Git :
```shell
git clone https://github.com/Scientest23/doc-classifier
cd doc-classifier
```

Créer un environnement python :
```shell
# Windows:
python -m venv .venv
.venv\Scripts\activate.bat

# Linux:
python3 -m venv .venv
source .venv/bin/activate
```

Requirements :
```shell
# Update pip:
pip install --update pip

# Install requirements:
pip install -r requirements.txt
```

## 📄 Datasets

Name | Size | Images | Download
-----|------|--------|-----------
Text extraction for OCR| 33.5 Mo | 520 | [Google Drive](https://drive.google.com/file/d/1w0FhoxyHAjFrWJBQ63JiJFPBsxBEUOVO/view?usp=drive_link)
Projet OCR classification | 203.3 Mo | 1 608 | [Google Drive](https://drive.google.com/file/d/1wDa-pXwdUmEpubo8UjGwQCWzZt9PNZbI/view?usp=drive_link)
RVL-CDIP Dataset (5% of original dataset)| 1.8 Go | 20 000 | [Google Drive](https://drive.google.com/file/d/13V-hUMebr5PZjqNcuwxUPlB6u7lEq_gB/view?usp=drive_link)
PRADO | 139.6 Mo | 1 589 | [Google Drive](https://drive.google.com/file/d/1Seii3yeWKoc4f9eUNcgAwDtFleO0yjUm/view?usp=drive_link)
RVL-CDIP Dataset | 38.8 Go | 400 000 | [Hugging Face](https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz?download=true)


To extract and import datasets, see the notebook: [00-hryniewski-download-datasets.ipynb](notebooks/00-hryniewski-download-datasets.ipynb)


## 🙌 Authors:
- Bryan Fernandez
- Ayoub Benkou
- Achraf Asri
- Daniel Hryniewski