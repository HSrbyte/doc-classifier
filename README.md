# doc-classifier
Projet de classification de documents avec la formation DataScientest (novembre 2023).



## 🛠️ Installation

Download Git project:
```shell
git clone https://github.com/Scientest23/doc-classifier
cd doc-classifier

# User config :
git config user.name "YourName"
git config user.email "YourEmail"
```

Create python environnement :
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