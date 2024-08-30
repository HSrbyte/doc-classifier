import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from src import image_grid_sample, image_read
import os
import glob
import cv2


st.write("# Deuxième modélisation : Classification à partir des images.")

pages = ["Modélisation",
         "Analyses des résultats", "Interprétation des modèles"]
page = st.sidebar.radio("Aller vers", pages)

models_list = ["CNN", "SqueezeNet",
               "MobileNetV2", "EfficientNetB1", "ResNet50"]

labels = ["Email",
          "Handwritten",
          "Invoice",
          "ID Card",
          "Passeport",
          "Scientific publication"]


# =========================================================================================================
#                                              PAGE 0
# =========================================================================================================
if page == pages[0]:

    st.header("Protocole d'entraînement des modèles")
    st.divider()

    st.header("Chargement des images")
    # Voir image comparison complement streamlit pour montrer une image avant et après le passage dans la moulinette
    st.divider()

    st.header("Choix des modèles")


# =========================================================================================================
#                                              PAGE 1
# =========================================================================================================
if page == pages[1]:

    options = st.sidebar.multiselect(
        "Choix des modèles", models_list, models_list)

    if options:
        st.header("Les courbes d'entraînements")

        def plot_val():

            df_loss = pd.DataFrame()
            df_acc = pd.DataFrame()
            df_loss["Epoch"] = range(1, 100 + 1)
            df_acc["Epoch"] = df_loss["Epoch"]

            for name in options:
                file_path = f"models/{name}_history.json"

                with open(file_path, "r") as file:
                    data = json.load(file)

                df_loss[name] = data["val_loss"]
                df_acc[name] = data["val_accuracy"]

            fig_loss = px.line(df_loss, x="Epoch", y=options)
            fig_loss.update_layout(
                title={
                    'text': "Comparaison des Loss-Validation en fonction des Epochs ",
                    'y': 0.92,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis_title="Epochs",
                yaxis_title="Loss",
                legend_title="Modèles"
            )

            fig_acc = px.line(df_acc, x="Epoch", y=options)
            fig_acc.update_layout(
                title={
                    'text': "Comparaison des Accuracy-Validation en fonction des Epochs",
                    'y': 0.92,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis_title="Epochs",
                yaxis_title="Accuracy",
                legend_title="Modèles")

            st.plotly_chart(fig_loss)
            st.plotly_chart(fig_acc)

        plot_val()
        st.divider()

        st.header("Les matrices de confusions")
        st.write('''
                Test text
            ''')
        label_axis = ["Email", "Handwritten", "Invoice",
                      "ID Card", "Passeport", "Scientific publication"]

        def confusion_matrix_display(options, label_axis):
            # Create tabs based on the selected models
            tab_model = st.tabs(options)

            # Iterate through each option and corresponding tab
            for i, name in enumerate(options):
                with tab_model[i]:
                    # Define the file path for the confusion matrix CSV
                    # file_path = f"results/confusion_matrix_{name}.csv"
                    file_path = f'references\{name}\confusion_matrix_{name}'

                    # Read the CSV into a DataFrame
                    df = pd.read_csv(file_path)

                    # Create a heatmap using Plotly
                    fig = px.imshow(df,
                                    labels=dict(x="Classes prédites",
                                                y="Classes réelles"),
                                    x=label_axis,
                                    y=label_axis,
                                    text_auto=True,
                                    aspect="auto"
                                    )
                    fig.update_xaxes(side="bottom")

                    # Display the chart in Streamlit
                    st.plotly_chart(fig)

        confusion_matrix_display(options, label_axis)

        st.divider()

        st.header("Précisions des modèles")
        st.write('''
                    Test text
                ''')
        df = pd.read_csv(
            r"results\category_accuracies_summary.csv", index_col=0)
        df = df.loc[options]

        # Set the index to the first column (document types) and transpose the DataFrame

        # Reset index to use document types as a column
        df = df.reset_index()
        df = df.melt(id_vars='index', var_name='Document Type',
                     value_name='Score')
        df.columns = ['Model', 'Document Type', 'Score']

        # Create the interactive bar chart using Plotly
        fig = go.Figure()

        for model in df['Model'].unique():
            fig.add_trace(go.Bar(
                x=df[df['Model'] == model]['Document Type'],
                y=df[df['Model'] == model]['Score'],
                name=model,
                hovertemplate="<b>Score:</b> %{y:.3f}<extra></extra>"
            ))

        fig.update_layout(
            barmode='group',
            title='Précision des modèles en fonction du type de document',
            xaxis_title='Type de document',
            yaxis_title='Précision',
            xaxis_tickangle=-45,
            yaxis=dict(
                    range=[0.85, 1]  # Set y-axis range from 0.85 to 1
            ),
            legend_title='Modèles'
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)
        st.divider()

        st.header("Vitesse de prédicion")
        st.markdown('''
                    Voici les caractéristiques du PC ayant réalisé les tests :
                    - Cartegraphique : NVIDIA GeForce RTX 2070 with Max-Q
                    - Processeur : Intel® Core™ i7-10750H CPU @ 2.60 GHz
                    - Capacité mémoire de la carte graphique : 8192 MiB
                    - Utilisation de la mémoire pendant les tests : 365 MiB
                    - Version dupilote NVIDIA : 522.06
                    - Version CUDA:11.8
                ''')

        df_perf = pd.read_csv(
            "results\prediction_performance.csv", index_col=0)
        st.dataframe(df_perf.loc[options])

    else:
        st.markdown("Choisir au moins un modèle")


# =========================================================================================================
#                                              PAGE 2
# =========================================================================================================
if page == pages[2]:

    options = st.sidebar.selectbox(
        "Choix du modèle", models_list)

    selected_doc = st.sidebar.selectbox(
        "Choix du type de document à analyser", options=labels)

    if options:
        st.header(
            f'Analyse des erreurs de prédiction du modèle {options} pour la classe "{selected_doc}" :')

        df_wrong_pred = pd.read_csv(
            f"references\{options}\wrong_predictions_{options}")

        subset = df_wrong_pred[df_wrong_pred["Classe réelle"] == selected_doc]

        if len(subset) > 0:  # Proceed only if the subset is not empty
            n_samples = min(8, len(subset))
            wrong_images = list(
                subset["image_path"].sample(n=n_samples, random_state=42))

            if len(wrong_images) >= 4:
                n = 4
            else:
                n = len(wrong_images)

            groups = []
            for i in range(0, len(wrong_images), n):
                groups.append(wrong_images[i:i+n])

            cols = st.columns(n)
            for group in groups:
                for i, image_file in enumerate(group):
                    real_category = df_wrong_pred["Classe réelle"][df_wrong_pred["image_path"]
                                                                   == image_file].values
                    pred_category = df_wrong_pred["Classe prédite"][df_wrong_pred["image_path"]
                                                                    == image_file].values
                    img = image_read(image_file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cols[i].image(
                        image_grid_sample(
                            [img], 1, 1, square_size=600, img_layout='center', seed=42), caption=f"Prédiction : {pred_category}")

        st.divider()

        st.header("Grad-CAM")
        st.markdown('Images originales')

        # Define the directory and pattern for the files
        file_pattern_originale = fr'references\{options}\originals\original_{options}_{selected_doc}_*.jpg'
        # List all files matching the pattern
        matching_files_originales = glob.glob(file_pattern_originale)

        groups_originales = []
        for a in range(0, len(matching_files_originales), 4):
            groups_originales.append(matching_files_originales[a:a+4])

        cols_originales = st.columns(4)
        for group in groups_originales:
            for a, image_file in enumerate(group):
                img_a = image_read(image_file)
                img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
                cols_originales[a].image(
                    image_grid_sample(
                        [img_a], 1, 1, square_size=600, img_layout='center', seed=42))

# -------------------------------------HEATMAPS--------------------------------------

        st.markdown("Image GradCam")
        # Define the directory and pattern for the files
        file_pattern_heatmap = fr'references\{options}\heatmaps\heatmap_{options}_{selected_doc}_*.jpg'
        # List all files matching the pattern
        matching_files_heatmaps = glob.glob(file_pattern_heatmap)

        groups_heatmaps = []
        for b in range(0, len(matching_files_heatmaps), 4):
            groups_heatmaps.append(matching_files_heatmaps[b:b+4])

        cols_heatmaps = st.columns(4)
        for group in groups_heatmaps:
            for b, image_file in enumerate(group):
                img_b = image_read(image_file)
                img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
                cols_heatmaps[b].image(
                    image_grid_sample(
                        [img_b], 1, 1, square_size=600, img_layout='center', seed=42))

    else:
        st.markdown("Choisir au moins un modèle")
