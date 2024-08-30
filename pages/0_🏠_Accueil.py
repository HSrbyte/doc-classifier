import streamlit as st
from PIL import Image


# Configuration de la page
st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")

# Titre de l'application
st.title("DOC-CLASSIFIER")
# Création des tabs
tabs = st.tabs(["**PRÉSENTATION**", "**PHASES DE CONCEPTION**", "**ÉQUIPE DE PROJET**"])

# Contenu pour le tab "Introduction"
with tabs[0]:
    st.subheader("""
    _Solution conçue pour simplifier la classification des documents!_ 
    En associant l’analyse du texte par apprentissage automatique _(Machine Learning)_ et le traitement des images par réseaux de neurones _(Deep Learning)_,
    doc-classifier accélère et automatise la reconnaissance des types de documents.

    Fiable, simple et efficace, il s'intègre facilement dans les systèmes existants et améliore les processus d'archivage et de recherche dans divers secteurs :

                
    * _Services gouvernementaux_
    * _Entreprises de gestion documentaire_
    * _Institutions financières et bancaires_
    * _Services d'archivage_
    * ...

    """)
    


    # Chemin vers votre fichier PNG
    image_path = "references/illustration1.png"


    st.image(image_path, caption='Présentation de Doc-Classifier', width=1300)




    # Ajouter une séparation stylisée
    st.markdown("---")
    st.info("""
    ### _Fonctionnalités de l'application_
    - **Classification automatique** de documents textuels et visuels.
    - **Choix du modèle** basé sur le type de contenu : _texte_, _image_, ou _les deux_.
    - **Traitement en temps réel** des données entrantes.
    """)

# Contenu pour le tab "Déroulement du Projet"
with tabs[1]:
    st.markdown("""
    Ce projet, développé lors de la formation _DataScientest_ (novembre 2023), se divise en trois phases clés :
    """)

    # Accordéon pour les étapes du projet
    with st.expander("1 - Analyse exploratoire des datasets"):
        st.markdown("""
        > Cette étape inclut la compréhension et l'analyse initiale des jeux de données avec des visualisations et des statistiques descriptives. 
        Elle prépare le terrain pour les étapes suivantes.
        
        ***Notebooks 00 à 05***
        """)
        

    with st.expander("2 - Première modélisation : Classification des données textuelles"):
        st.markdown("""
        > Utilisation de Tesseract pour l'extraction de texte, suivie d'une classification basée sur les données textuelles. 
        Les modèles développés sont axés sur le traitement du langage naturel (NLP).
                    
        ***Notebooks 06 à 11***
        """)

    with st.expander("3 - Deuxième modélisation : Classification des images"):
        st.markdown("""
        > Application de techniques de Deep Learning pour classifier les documents basés sur les images. 
        Cette étape inclut le prétraitement des images, la conception de modèles CNN, et l'évaluation des performances.
                    
        ***Notebooks 12 à 20***
        """)

    st.caption("""
    _Pour plus de détails, veuillez consulter les notebooks correspondants.
        Chaque étape est documentée en profondeur pour vous guider tout au long du projet._
    """)

# Contenu pour le tab "Équipe et Infos"
with tabs[2]:
    st.markdown("#### Équipe de Projet")

    # Mise en page avec des colonnes pour afficher les noms des auteurs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""\n 
                    - Achraf Asri 
                    - Ayoub Benkou
                    - Bryan Fernandez
                    - Daniel Hryniewski""")

    st.markdown("---")

    st.markdown("#### Management")

    # Mise en page avec des colonnes pour afficher les noms des auteurs
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown("""\n 
                    - Sebastien S
                    \n - Lisa B""")

    with col3:
        st.markdown("""\n
                     ***Project Manager*** 
                     \n***Cohorte Chief***""")

    with col4:
        st.image("https://datascientest.com/wp-content/uploads/2020/08/new-logo.png", width=20)
        st.image("https://datascientest.com/wp-content/uploads/2020/08/new-logo.png", width=20)

    # Afficher un séparateur ou un espace pour structurer la page
    st.markdown("---")

    st.markdown(
        """*More on doc-classifier* :
        <a href="https://github.com/Scientest23/doc-classifier" target="_blank" rel="noopener noreferrer">
            <img alt="GitHub" src="https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github">
        </a>
        """,
        unsafe_allow_html=True
    )