import pandas as pd
import streamlit as st
import random
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import cv2
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objects as go
import os
import numpy as np
import plotly.figure_factory as ff

# Chargement du DataFrame
df_images = pd.read_csv('data/image_info_labeled.csv')

# Ajouter une colonne pixels pour la distribution des pixels
df_images['pixels'] = df_images['height'] * df_images['width']

# Transformation des chemins relatifs en URL
df_images['path'] = df_images['path'].apply(lambda x: x.replace('\\', '/'))

# Fonction pour afficher les échantillons d'images avec les informations
@st.cache_data(ttl=0)
def display_image_samples_with_details(df, dataset_name):
    # Filtrer les images pour le dataset spécifique
    df_filtered = df[df['dataset'] == dataset_name]
    
    # Prendre un échantillon de 6 images
    sample_df = df_filtered.sample(n=8)

    # Créer une disposition de 2 lignes x 3 colonnes
    rows = [st.columns(4) for _ in range(2)]

    for idx, (row_idx, row) in enumerate(sample_df.iterrows()):
        # Calculer la ligne et la colonne où l'image doit être placée
        row_num = idx // 4  
        col_num = idx % 4
        
        with rows[row_num][col_num]:
            # Afficher l'image avec les informations en légende
            st.image(row['path'], caption=f"{row['image_name']} - {row['label']} - {row['width']}x{row['height']} - {row['colorspace']}", use_column_width=True)

# Fonction pour calculer l'histogramme des couleurs d'une image
def compute_color_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calcul de l'histogramme pour chaque canal (R, G, B)
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Normalisation des histogrammes
    hist_r = hist_r / hist_r.sum()
    hist_g = hist_g / hist_g.sum()
    hist_b = hist_b / hist_b.sum()

    return hist_r.flatten(), hist_g.flatten(), hist_b.flatten()



# Titre principal de la page
st.title("Analyse exploratoire des données")

# Création des onglets
tab1, tab2, tab3 = st.tabs(["SOURCES DE DONNÉES", "ANALYSE EXPLORATOIRE ET VISUALISATION", "ANALYSE DES DONNÉES TEXTUELLES (OCR)"])


#_______________________________________________________________________________________________________________________________________

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

# Contenu du premier onglet: Sources de données
with tab1:
    st.header("Sources de données")
    st.write("Les données brutes utilisées dans le cadre de ce projet proviennent des 5 sources suivantes :")

    # Données du tableau
    data = {
        "Dataset": [
            "1",
            "2",
            "3",
            "4",
        ],
        "Name": [
            "Text extraction for OCR",
            "Projet OCR classification",
            "RVL-CDIP Dataset",
            "PRADO",
        ],
        "Size": [
            "33.5 Mo",
            "203.3 Mo",
            "38.8 Go",
            "139.6 Mo",
        ],
        "Images": [
            520,
            1_608,
            400_000,
            1_589,
        ],
        "Download": [
            "https://drive.google.com/file/d/1w0FhoxyHAjFrWJBQ63JiJFPBsxBEUOVO/view?usp=drive_link",
            "https://drive.google.com/file/d/1wDa-pXwdUmEpubo8UjGwQCWzZt9PNZbI/view?usp=drive_link",
            "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz?download=true",
            "https://drive.google.com/file/d/13V-hUMebr5PZjqNcuwxUPlB6u7lEq_gB/view?usp=drive_link",
        ],
    }

    # Création du DataFrame
    df = pd.DataFrame(data)

    # Descriptions des datasets
    descriptions = {
        "Text extraction for OCR": "L'ensemble de données est constitué de fichiers XML et d'images. Les fichiers XML contiennent les données textuelles extraites des images. Ce dataset comprend au total 519 images de factures.",
        "Projet OCR classification": "Le dataset contient 1607 images de divers types de documents répartis en 23 classes.",
        "PRADO": "Ce dataset contient des spécimens de passeports et de cartes nationales d'identité provenant de divers pays du monde, ajoutés de notre propre initiative. Les données proviennent du Conseil européen.",
        "RVL-CDIP Dataset": "L'ensemble de données RVL-CDIP se compose de 400 000 images en niveaux de gris réparties en 16 classes, avec 25 000 images par classe. Les images sont redimensionnées pour ne pas dépasser 1 000 pixels."
    }

    # Ajout des descriptions dans le DataFrame
    df["Description"] = df["Name"].map(descriptions)

    # Affichage du tableau avec les colonnes configurées
    st.data_editor(
        df,
        column_config={
            "Download": st.column_config.LinkColumn(
                "Téléchargement",
                help="Cliquez pour télécharger les données",
                display_text="Ouvrir le lien"
            )
        },
        hide_index=True,
    )

    # Caption sous le tableau
    st.caption(
        """ _**Remarque** : Dans le cadre du projet Doc-Classifier, nous avons utilisé seulement 5% du volume total du dataset 
        RVL-CDIP. Pour améliorer les performances des modèles, une perspective future serait de les réentraîner 
        sur l'ensemble complet du dataset RVL-CDIP._"
    """)

    st.markdown("---")
    st.header("Visualisation des échantillons d'images")

    st.write("Choisissez un dataset pour afficher un échantillon d'images :")

    # Liste des datasets disponibles dans le DataFrame
    datasets = df_images['dataset'].unique()

    # Créer des expanders pour afficher les échantillons de chaque dataset
    for dataset in datasets:
        
        with st.expander(f"Echantillons du dataset : {dataset}"):
            
            # Recharger un nouvel échantillon à chaque ouverture de l'expander
            
            display_image_samples_with_details(df_images, dataset)
    
    st.markdown("---")
    st.header("Répartition des images par catégorie dans les jeux de données")


    # Uniformisation des catégories
    df_images['label'] = df_images['label'].replace({
        'facture': 'invoice', 
        'Facture': 'invoice', 
        'Invoice': 'invoice',
        'id_pieces': 'id_pieces', 
        'national_identity_card': 'id_pieces'
    })

    # Compter le nombre d'images par catégorie dans chaque dataset
    counts = df_images.groupby(['label', 'dataset']).size().reset_index(name='count')

    # Créer un graphique interactif avec Plotly
    fig = px.bar(
        counts,
        x='label',
        y='count',
        color='dataset',
        labels={'label': "Catégories d'images", 'count': "Nombre d'images"},
        category_orders={'label': sorted(df_images['label'].unique())}
    )

    # Afficher le graphique interactif dans Streamlit
    st.plotly_chart(fig)

    st.markdown(""" > #####  *Récaptilatif des observations* :
> - **Incohérences de classement entre les catégories** : Présence de CNI dans la catégorie « passeport » et des factures dans la catégorie "justif_domicile".
> - **data_01** : uniquement composé de factures.
> - **data_02** : montre un déséquilibre entre les catégories, ce dataset n'a pas été utilisé dans la suite du projet.
> - **data_03** : équilibré avec 5% d'images par catégorie.
> - **data_04** : contient uniquement des documents d'identité, avec une prédominance de passeports.
> - Les catégories ont été uniformisées pour assurer la cohérence dans l'analyse. Par exemple, toutes les variantes de _facture_ ont été regroupées sous _invoice_, et les catégories liées aux pièces d'identité sous _id_pieces_.""")


    st.markdown("---")

#_______________________________________________________________________________________________________________________________________


# Contenu du deuxième onglet: Analyse exploratoire et visualisation
with tab2:
    st.header("Visualisation des caractéristiques des images par dataset")
    # 1. Distribution des Dimensions des Images
    with st.expander("I - Dimensions des images", expanded=False):
        st.subheader("I - Distribution des dimensions")
        
        # Nuage de points interactif
        scatter_fig = px.scatter(df_images, x='width', y='height', color='dataset',
                                title="Hauteur vs Largeur des Images",
                                labels={'width': 'Largeur', 'height': 'Hauteur'})
        st.plotly_chart(scatter_fig)

        # Boxplot interactif des hauteurs
        boxplot_fig = px.box(df_images, x='dataset', y='height',
                            title="Distribution des hauteurs par dataset",
                            labels={'dataset': 'Dataset', 'height': 'Hauteur'})
        st.plotly_chart(boxplot_fig)
    
    #_______________________________________________________________________________________________________________________________________


    # 2. Analyse de la Netteté et du Flou des Images
    with st.expander("II - Netteté et flou des images"):
        st.subheader("II - Netteté et flou")

        # Violin plot interactif du flou avec options réduites
        violin_fig = px.violin(df_images, x='dataset', y='bluriness', box=True, points=False,
                            title="Distribution du flou par dataset",
                            labels={'dataset': 'Dataset', 'bluriness': 'Flou'})
        violin_fig.update_traces(hoverinfo='skip')  # Désactiver les tooltips
        st.plotly_chart(violin_fig)

        # Histogramme interactif du flou avec options réduites
        hist_blur_fig = px.histogram(df_images, x='bluriness', color='dataset', nbins=30,  # Réduire le nombre de bins
                                    title="Histogramme du Flou par Dataset",
                                    labels={'bluriness': 'Flou', 'count': 'Nombre d\'images'})
        hist_blur_fig.update_traces(hoverinfo='skip')  # Désactiver les tooltips
        st.plotly_chart(hist_blur_fig)

        # Filtrer les 10 images les plus floues par dataset
        # Filtrer les 10 images les plus floues par dataset
        top_blur_images = df_images.groupby('dataset').apply(lambda x: x.nlargest(10, 'bluriness')).reset_index(drop=True)

    #_______________________________________________________________________________________________________________________________________

    # 3. Répartition des Catégories d'Images
    with st.expander("III - Répartition des catégories d'images"):
        st.subheader("III - Répartition des catégories")

        # Préparer les données pour un diagramme en barres empilées
        counts = df_images['label'].value_counts().reset_index()
        counts.columns = ['label', 'count']

        # Diagramme en barres empilées
        stacked_bar_fig = px.bar(
            counts, x='label', y='count',
            title="Répartition des Images par Catégorie",
            labels={'label': 'Catégorie', 'count': 'Nombre d\'images'},
            color='label',
            text='count'
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(stacked_bar_fig)

        # Distribution des pixels par catégorie avec boxplot interactif
        boxplot_pixels_fig = px.box(df_images, x='label', y='pixels', color='dataset',
                                    title="Distribution des Pixels par Catégorie",
                                    labels={'label': 'Catégorie', 'pixels': 'Pixels (Hauteur x Largeur)'})
        st.plotly_chart(boxplot_pixels_fig)

        st.info("""
            L'analyse des graphiques a permis de sélectionner 6 classes parmi 22, en se basant sur leur représentation
            suffisante dans les datasets : 

            - Passeport
            - Carte d'identité
            - Email
            - Facture
            - Publication scientifique*
            """)
    
    #_______________________________________________________________________________________________________________________________________


    # 4. Analyse des Couleurs et de la Nature des Images
    with st.expander("VI - Analyse des couleurs et de la nature des images", expanded = True):
        st.subheader("VI - Couleurs et nature des images")

        # Liste des datasets disponibles (data_01, data_02, ...)
        datasets = ["data_01", "data_02", "data_03", "data_04"]
        
        # Sélection du dataset
        dataset_choice = st.selectbox("Choisissez un Dataset", datasets)

        # Chemin vers le dossier contenant les images du dataset choisi
        image_folder = os.path.join("data", "raw", dataset_choice, "images")

        # Liste pour stocker tous les histogrammes
        all_histograms_r = []
        all_histograms_g = []
        all_histograms_b = []

        # Parcourir toutes les images du dossier sélectionné
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            hist_r, hist_g, hist_b = compute_color_histogram(image_path)

            all_histograms_r.append(hist_r)
            all_histograms_g.append(hist_g)
            all_histograms_b.append(hist_b)

        # Moyenne des histogrammes
        avg_hist_r = np.mean(all_histograms_r, axis=0)
        avg_hist_g = np.mean(all_histograms_g, axis=0)
        avg_hist_b = np.mean(all_histograms_b, axis=0)

        # Création du graphique avec Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(y=avg_hist_r, mode='lines', name='Rouge', line=dict(color='red')))
        fig.add_trace(go.Scatter(y=avg_hist_g, mode='lines', name='Vert', line=dict(color='green')))
        fig.add_trace(go.Scatter(y=avg_hist_b, mode='lines', name='Bleu', line=dict(color='blue')))

        fig.update_layout(title=f'Histogramme des Couleurs - Dataset {dataset_choice}',
                        xaxis_title='Intensité de couleur',
                        yaxis_title='Fréquence',
                        legend_title="Canal de Couleur")

        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig)

    # Footer
    st.markdown("""
    > #### *_Conclusion_*
    > Ces visualisations offrent une vue d'ensemble des caractéristiques principales des datasets, 
    notamment les dimensions, la netteté, la répartition des catégories, et la nature des images. 
    Ces insights sont essentiels pour orienter les prochaines étapes de la préparation des données 
    et de la construction du modèle de classification.
    """)

#_______________________________________________________________________________________________________________________________________


# Contenu du troisième onglet: Analyse des données textuelles (OCR)
with tab3:
    st.header("Analyse des données textuelles (OCR)")

    # Chargement des jeux de données textuelles
    df1 = pd.read_csv("data/raw/data_01/text_process.csv")
    df2 = pd.read_csv("data/raw/data_02/text_process.csv")
    df3 = pd.read_csv("data/raw/data_03/text_process.csv")
    df4 = pd.read_csv("data/raw/data_04/text_process.csv")

    # Préparation des données
    for df in [df1, df2, df3, df4]:
        df['words'] = df['words'].apply(lambda x: x.split(' ') if isinstance(x, str) else [])
        df['words_count'] = df['words'].apply(lambda x: len(x))

    df2['category'] = df2['category'].str.replace('facture', 'invoice')

    # Ajout d'une colonne 'dataset' pour chaque dataframe
    df1['dataset'] = 'data_01'
    df2['dataset'] = 'data_02'
    df3['dataset'] = 'data_03'
    df4['dataset'] = 'data_04'

    # Concatenation des datasets
    df = pd.concat([df1, df2, df3, df4])


    #_______________________________________________________________________________________________________________________________________

    # Contenu du troisième onglet: Analyse des données textuelles (OCR)
    with st.expander("Analyse linguistique", expanded= True):
        st.subheader("Distribution des langues dans l'ensemble des jeux de données")
        lang_count = df['lang'].value_counts()
        fig_lang = px.bar(x=lang_count.index, y=lang_count.values, labels={'x': 'Langue', 'y': 'Nombre d\'occurrences'},
                        )
        st.plotly_chart(fig_lang)
        st.info("La langue anglaise est la plus présente dans notre jeu de données global représentant 75.44%.")


    #_______________________________________________________________________________________________________________________________________

    with st.expander("Nombre de mots par catégorie"):
        st.subheader("Distribution du nombre de mots par page par catégorie")

        # Sélectionner un dataset pour afficher les graphiques correspondants
        selected_dataset = st.selectbox("Choisissez un dataset", df['dataset'].unique())

        sub_data = df[df['dataset'] == selected_dataset]

        # Création d'un box plot pour visualiser la distribution du nombre de mots par page
        fig_words = px.box(sub_data, x='category', y='words_count', color='category',
                        title=f'Distribution du nombre de mots par page ({selected_dataset})',
                        labels={'category': 'Catégorie', 'words_count': 'Nombre de mots par page'},
                        color_discrete_sequence=px.colors.qualitative.Safe)

        st.plotly_chart(fig_words)

        st.write("""
        Ces graphiques montrent la distribution du nombre de mots par page dans chaque catégorie. 
        Les catégories comme "scientific_publication" et "news_article" contiennent plus de texte, 
        tandis que d'autres comme "passeport" en ont moins.
        """)

    #_______________________________________________________________________________________________________________________________________

    with st.expander("Top mots par catégorie"):
        st.subheader("Quels mots caractérisent chaque catégorie ?")

        # Sélectionner une catégorie pour afficher les mots les plus fréquents
        selected_category = st.selectbox("Choisissez une catégorie", df['category'].unique())

        df_cat = df[df['category'] == selected_category]
        word_freq = Counter(df_cat['words'].sum())
        top_words, frequencies = zip(*word_freq.most_common(10))  # Afficher les 10 mots les plus fréquents

        # Calculer les pourcentages
        total = sum(frequencies)
        percentages = [f"{(freq / total) * 100:.1f}%" for freq in frequencies]

        # Création d'un graphique en treemap avec des pourcentages
        fig_words_treemap = go.Figure(go.Treemap(
            labels=[f"{word} ({percent})" for word, percent in zip(top_words, percentages)],
            parents=[""] * len(top_words),  # Utilisation de "" pour que tous les mots soient au même niveau
            values=frequencies,
            textinfo="label",  # Afficher uniquement les labels (avec pourcentage)
            marker=dict(colors=frequencies, colorscale="Reds"),  # Coloration basée sur la fréquence
            textfont=dict(size=20),  # Augmenter la taille de la police
        ))

        fig_words_treemap.update_layout(
            title=f"Mots les plus fréquents dans la catégorie : {selected_category}",
        )

        st.plotly_chart(fig_words_treemap)

        st.write("""
        Les mots les plus fréquents sont souvent représentatifs du type de document analysé.
        """)

        # Ajouter un exemple de WordCloud
        if st.checkbox("Afficher un exemple de WordCloud pour cette catégorie"):
            st.subheader(f"WordCloud pour la catégorie : {selected_category}")

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_cat['words'].sum()))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)




