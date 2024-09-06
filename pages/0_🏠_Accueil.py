import streamlit as st
from PIL import Image


# Configuration de la page
st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")

# Titre de l'application
st.title("DOC-CLASSIFIER")

# Création des tabs
tabs = st.tabs(["**PRÉSENTATION**", "**PHASES DE CONCEPTION**"])

# Contenu pour le tab "Présentation"
with tabs[0]:
    


    st.markdown("""
        **Doc-Classifier** est une application conçue pour simplifier la classification des documents. 
        Elle rend la reconnaissance de documents rapide en combinant analyse textuelle et traitement d'images.""")

    st.info("""
        OBJECTIFS :
        > - **Automatiser la classification** : Faciliter l’organisation et la gestion des documents.
        > - **Intégration simple** : Fonctionner avec les workflows existants.
        > - **Optimiser les processus** : Améliorer la recherche et l’archivage des documents dans divers secteurs, tels que les _**services gouvernementaux**_, _**la gestion documentaire**_, _**les institutions financières**_, etc.
        """)




    # Image en dessous des colonnes
    image_path = "references/illustration4.png"
    st.image(image_path, caption='Présentation de Doc-Classifier', use_column_width=True)
    
    # Ajouter une séparation stylisée
    st.markdown("---")
    
    # Fonctionnalités de l'application en bas
    st.markdown("""
    FONCTIONNALITÉS DE L'APPLICATION :
    > - **Classification automatique** de documents textuels et visuels.
    > - **Choix du modèle** basé sur le type de contenu : _texte_, _image_, ou _les deux_.
    > - **Traitement en temps réel** des données entrantes et évaluation des performances.
    """)

# Contenu pour le tab "Déroulement du Projet"
with tabs[1]:

    st.markdown("""
    Ce projet, développé lors de la formation <span style='color: cyan;'>_DataScientest_</span> _(novembre 2023)_, se divise en trois phases clés :
    """, unsafe_allow_html=True)

    # Accordéon pour les étapes du projet
    with st.expander("1 - Analyse exploratoire des datasets"):
        st.markdown("""
        > Cette étape inclut la compréhension et l'analyse initiale des jeux de données avec des visualisations et des statistiques descriptives. 
        Elle prépare le terrain pour les étapes suivantes.
        
        ***Notebooks 00 à 05***
        """)
        
    with st.expander("2 - Première modélisation : Classification des données textuelles"):
        st.markdown("""
        > Extraction des données textuelles des documents, suivie d'une classification basée sur leur contenu. 
        Les modèles développés sont axés sur le traitement du langage naturel.
                    
        ***Notebooks 06 à 11***
        """)

    with st.expander("3 - Deuxième modélisation : Classification des images"):
        st.markdown("""
        > Application de techniques du *_DeepLearning_* pour classifier les documents basés sur les images. 
        Cette étape inclut le prétraitement des images, la conception des réseaux de neurones et l'évaluation des performances.
                    
        ***Notebooks 12 à 20***
        """)

    st.caption("""
    _Pour plus de détails, veuillez consulter les notebooks correspondants.
        Chaque étape est documentée en profondeur pour vous guider tout au long du projet._
    """)

# Contenu de l'équipe déplacé dans la sidebar
st.sidebar.markdown("### Équipe de Projet")

# Mise en page avec des colonnes pour afficher les noms des auteurs dans la sidebar
st.sidebar.markdown("""\n 
                - Achraf Asri 
                - Ayoub Benkou
                - Bryan Fernandez
                - Daniel Hryniewski""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Management")

# Affichage des détails du management dans la sidebar
# Mise en page avec des colonnes pour afficher les noms et les logos sur la même ligne

st.sidebar.markdown("""
    - Sebastien S - _Project Manager_
    - Lisa B - _Cohorte Chief_
    """)

st.sidebar.markdown("---")

# Lien GitHub dans la sidebar
st.sidebar.markdown(
    """*More on doc-classifier* :
    <a href="https://github.com/Scientest23/doc-classifier" target="_blank" rel="noopener noreferrer">
        <img alt="GitHub" src="https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github">
    </a>
    """,
    unsafe_allow_html=True
)