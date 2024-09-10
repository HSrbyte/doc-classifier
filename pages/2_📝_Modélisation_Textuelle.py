import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Fonction pour charger les données JSON
def st_read_jsonfile(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Fonction pour afficher les résultats de la recherche avec Plotly en sous-graphes
def st_plot_search_results(cv_results: dict, best_params: dict, param_grid: dict, title: str) -> None:
    """Plots the search results from a cross-validation grid search using Plotly subplots.

    Args:
        cv_results (dict): The cross-validation results containing mean and
                           standard deviation of test scores for different parameter
                           combinations. This is typically the `cv_results_`
                           attribute of a fitted GridSearchCV object.
        best_params (dict): The best parameters found by the grid search. This
                            corresponds to the `best_params_` attribute of a
                            fitted GridSearchCV object.
        param_grid (dict): The parameter grid used for the grid search. This is
                           the same as the parameter grid passed to the GridSearchCV.
        title (str): The title for the plot.
    """
    # Convert cv_results to DataFrame if it is not already
    if not isinstance(cv_results, pd.DataFrame):
        cv_results = pd.DataFrame(cv_results)

    masks_names = [k for k, v in param_grid.items() if len(v) > 1]
    plot_results = {}
    for pk, pv in best_params.items():
        if pk not in masks_names:
            param_grid.pop(pk)
            continue
        plot_results[pk] = [[], [], []]
        for val in param_grid[pk]:
            if val is not None:
                res_param = cv_results[cv_results[f'param_{pk}'] == val]
            else:
                res_param = cv_results.loc[cv_results[f'param_{pk}'].isnull()]

            id_ = res_param['mean_test_score'].idxmax()

            if pd.isna(id_):
                plot_results[pk][0].append(str(val))
                plot_results[pk][1].append(0.0)
                plot_results[pk][2].append(0.0)
            else:
                row = cv_results.iloc[id_]
                mean_test_score = row['mean_test_score']
                std_test_score = row['std_test_score']
                plot_results[pk][0].append(str(val))
                plot_results[pk][1].append(mean_test_score)
                plot_results[pk][2].append(std_test_score)

    # Create subplots using Plotly
    num_plots = len(plot_results)
    fig = make_subplots(
        rows=1, cols=num_plots,
        subplot_titles=[name.upper() for name in plot_results.keys()],
        horizontal_spacing=0.05
    )

    # Add traces for each parameter in subplots
    for i, (name, values) in enumerate(plot_results.items(), start=1):
        x = np.array(values[0])
        y = np.array(values[1])
        e = np.array(values[2])
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=dict(type='data', array=e, visible=True),
                mode='lines+markers',
                name=name.upper(),
                line=dict(dash='dash')
            ),
            row=1, col=i
        )

    # Update layout
    fig.update_layout(
        title_text=title + '<br>Score per parameter',
        xaxis_title='Parameter Value',
        yaxis_title='Mean Score',
        template='plotly_dark',
        showlegend=True
    )

    st.plotly_chart(fig)

st.write("# Première modélisation : Classification à partir des données textuelles.")

# Onglets principaux
tabs = st.tabs(["Préambule", "Choix des Modèles", "Résultats et Conclusions"])

# Onglet Introduction et Prétraitement des Données
with tabs[0]:
    st.header("Introduction et prétraitement des données") 
    st.write("""
    - **Présentation de l'objectif :** « Le but est de classifier les documents en attribuant la bonne catégorie en se basant uniquement sur les données textuelles extraites des images. »
    - **Variable cible :** « La variable à prédire est la catégorie (category) des documents. »
    
    **Méthodologie de Prétraitement:**
    - **Extraction du Texte:** « Le texte des images a été extrait en utilisant Tesseract, un outil OCR (reconnaissance optique de caractères). »
    """)
    
    # Affichage de l'image
    image_path = "references/preparation_donnees_textuelle.png"
    st.image(image_path, caption="Préparation des données textuelles", use_column_width=True)
    # Initialiser les variables de session pour chaque DataFrame
    if "show_df_train" not in st.session_state:
        st.session_state.show_df_train = False

    if "show_df_test" not in st.session_state:
        st.session_state.show_df_test = False
    
    if "show_json" not in st.session_state:  
        st.session_state.show_json = False
    
    st.write("**Dataframes :**")

    # Bouton pour afficher/cacher le DataFrame `words_structure_train`
    if st.button("words_structure_train"):
        st.session_state.show_df_train = not st.session_state.show_df_train

    # Si `show_df_train` est True, afficher le DataFrame
    if st.session_state.show_df_train:
        df_train = pd.read_csv("data/processed/words_structure_train.csv")
        st.dataframe(df_train.head(5))

    # Bouton pour afficher/cacher le DataFrame `words_structure_test`
    if st.button("words_structure_test"):
        st.session_state.show_df_test = not st.session_state.show_df_test

    # Si `show_df_test` est True, afficher le DataFrame
    if st.session_state.show_df_test:
        df_test = pd.read_csv("data/processed/words_structure_test.csv")
        st.dataframe(df_test.head(5))
    
    # Bouton pour afficher/cacher le contenu du JSON `most_common_words.json`
    if st.button("most_common_words"):
        st.session_state.show_json = not st.session_state.show_json

    # Si `show_json` est True, afficher le contenu du JSON
    if st.session_state.show_json:
        with open("data/processed/most_common_words.json", "r") as file:
            data_json = json.load(file)
            st.json(data_json)  # Afficher le JSON dans un format lisible

    st.write("""
    - **Trois Approches:** « Trois méthodes de modélisation ont été envisagées : »
        - **Utilisation uniquement du corpus des pages. le dataframe sera appelé par la suite "words"** (Approche 1).
        - **Exploitation des données structurelles comme le nombre de mots et la diversité lexicale. Le dataframe sera appelé par la suite "structure"** (Approche 2).
        - **Combinaison des deux premières approches. Le dataframe sera appelé par la suite "Words & Structure"** (Approche 3).
    """)

# Onglet Modélisation et Choix des Modèles
with tabs[1]:
    st.header("Choix des Modèles")
    
    
    # Description des modèles de classification
    st.write("""
    **Choix des Modèles de Classification**

    Nous avons utilisé la librairie Python *Lazypredict* pour 
    faciliter la sélection des modèles de classification en 
    permettant une évaluation rapide avec un minimum de code. 
    L'objectif était d'identifier les modèles les plus performants
    pour prédire les catégories de documents à partir des données textuelles.

    Les modèles sélectionnés par *Lazypredict* comprennent :
    - Logistic Regression
    - Random Forest Classifier
    - Nearest Centroid
    - Extra Trees Classifier
    - XGBClassifier
    - LGBMClassifier
    """)

    # Affichage de l'image
    image_path = "references/lazypredict_workflow.png"
    st.image(image_path, caption="Lazypredict workflow", use_column_width=True)



# Liste des modèles disponibles
models = ['NearestCentroid', 'RandomForestClassifier', 'ExtraTreesClassifier', 'LGBMClassifier', 'LogisticRegression', 'XGBClassifier']

# Charger les résultats des modèles pour chaque type de données
data_results = {
    'structure': st_read_jsonfile('models/words_scaled_GridSearchCV_result.json'),
    'words': st_read_jsonfile('models/tfidfOnly_GridSearchCV_result.json'),
    'words & structure': st_read_jsonfile('models/words_structure_GridSearchCV_result.json')
}


with tabs[2] : 
    # Onglets pour organiser les résultats
    tabs = st.tabs(["GridSearchCV", "Best Models", "Conclusions"])


    # Onglet Grid Search CV
    with tabs[0]:
        st.header("Analyse des résultats de GridSearchCV")

        # Sélectionner le type de données via des boutons radio
        selected_type = st.radio(
            "Sélectionnez le type de données à afficher",
            options=['structure', 'words', 'words & structure']
        )

        # Afficher les graphiques pour le type de données sélectionné
        if selected_type:
            st.write(f"## Type de données: {selected_type}")
            
            # Afficher les graphiques pour chaque modèle
            for model_name in models:
                if model_name in data_results[selected_type]:
                    model_result = data_results[selected_type][model_name]
                    st.write(f"### Modèle: {model_name}")
                    st_plot_search_results(
                        model_result['cv_results'],
                        model_result['best_params'],
                        model_result['param_grid'],
                        model_name
                    )

    # Onglet Best Models
    with tabs[1]:
        st.header("Best Models")
        models.append('VotingClassifier')
        st.write("""
                 Après avoir finalisé et obtenu les meilleurs paramètres 
                 pour chaque modèle et chaque approche, nous avons combiné 
                 ces modèles de machine learning en un seul modèle de type 
                 Voting Classifier. Les résultats obtenus sont présentés 
                 dans le graphique ci-dessous.
                 """)

        # Sélectionner le type de données via un expander
        with st.expander("Choisissez le type de données", expanded=True):
            selected_type = st.selectbox("Sélectionnez le type de données à afficher", options=['structure', 'words', 'words & structure'])
        
        # Vérifier que le type de données est valide
        if selected_type not in data_results:
            st.error("Type de données non valide.")
        else:
            # Labels des catégories pour le type de données sélectionné
            categories = list(data_results[selected_type][models[0]]['accuracy'].keys())
            
            # Valeurs d'accuracy pour chaque modèle
            values = {}
            for model in models:
                if model in data_results[selected_type]:
                    values[model] = list(data_results[selected_type][model]['accuracy'].values())
                else:
                    values[model] = [0] * len(categories)  # Assigner des valeurs nulles si le modèle n'existe pas pour le type de données
            
            # Largeur des barres
            bar_width = 0.1

            # Positions des barres sur l'axe x pour chaque modèle
            positions = {model: [i + bar_width * idx for i in range(len(categories))] for idx, model in enumerate(models)}

            # Création du graphique avec Plotly
            fig = go.Figure()

            colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'pink']  # Couleurs pour les barres

            for idx, model in enumerate(models):
                fig.add_trace(go.Bar(
                    x=[i + bar_width * idx for i in range(len(categories))],
                    y=values[model],
                    name=model,
                    width=bar_width,
                    marker_color=colors[idx % len(colors)]
                ))

            # Mise en page
            fig.update_layout(
                title=f'Accuracy des modèles par catégorie pour {selected_type}',
                xaxis_title='Catégories',
                yaxis_title='Accuracy',
                barmode='group',
                xaxis=dict(
                    tickvals=[i + bar_width * (len(models) - 1) / 2 for i in range(len(categories))],
                    ticktext=categories
                ),
                yaxis=dict(range=[0.45, 1.1]),
                legend_title='Modèles',
                xaxis_title_font=dict(size=14, family='Arial, sans-serif'),
                yaxis_title_font=dict(size=14, family='Arial, sans-serif'),
                title_font=dict(size=16, family='Arial, sans-serif'),
                template='plotly_dark'
        )

        st.plotly_chart(fig)
        # Texte explicatif
        st.write("Voici le meilleur résultat d’accuracy des trois approches :")

        # Créer le tableau sous forme de markdown
        markdown_table = """
        | Approche                    | Meilleur modèle       | Accuracy |
        |-----------------------------|-----------------------|----------|
        | Structure des mots          | XGBClassifier         | 0.8185   |
        | Uniquement les mots         | LogisticRegression    | 0.8691   |
        | La combinaison des deux approches | VotingClassifier     | 0.8776   |
        """

        # Afficher le tableau dans Streamlit
        st.markdown(markdown_table)


# Onglet Conclusions (à remplir selon vos besoins)
with tabs[2]:
    st.header("Résultats et Conclusions")
    # Points clés
    st.write("""
    - **Modèle retenu :** Voting Classifier avec une accuracy de 87.8%
    - **Limitation des résultats :** 
        - Données biaisées
        - Images liées au tabac dans les catégories "email", "invoice", 
             "scientific publication"
             """)
    # Affichage de l'image
    image_path= "references/nuage_des_mots_pour_publication_scientifique.png"
    st.image(image_path, caption="Nuage des mots pour publication scientifique", use_column_width=True)

    # Affichage de l'image
    image_path = "references/nuage_des_mots_pour_facture.png"
    st.image(image_path, caption="Nuage des mots pour facture", use_column_width=True)
    
    st.write("""
    - **Impact :** 
        - Haute performance dans le domaine spécifique
        - Moins efficace dans d'autres contextes généraux
    """)
