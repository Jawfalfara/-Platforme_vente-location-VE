import streamlit as st
import warnings
warnings.filterwarnings('ignore')
# Import necessary libraries
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import plotly as py
import streamlit as st 
import query
import altair as alt
from streamlit_option_menu import option_menu 
from numerize.numerize import numerize 
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as plt
import streamlit_themes as st_theme
import warnings
warnings.filterwarnings('ignore')

#Streamlit Configuration 

#######################
# Page configuration
st.set_page_config(
    page_title="Plateforme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling


st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #1b2631;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stForm {
        background-color: #1b2631;
    }
    </style>
""", unsafe_allow_html=True)


#########################################
#DATA EXPLORER
#########################################
#import streamlit as st
import pandas as pd
import requests
import io



# Importer les données
df = pd.read_csv('IEA Global EV Data 2024.csv')
# Afficher les premières lignes du DataFrame
df.head()
# Afficher les informations générales
df.info()
# Afficher les statistiques descriptives
print(df.describe())
# valeurs nulles
df.isnull().sum()
#  supprimer les lignes avec des valeurs manquantes (if any)
df.duplicated()
df.drop_duplicates(inplace=True)

import streamlit as st

from gettext import install

import streamlit as st
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

import streamlit as st
import pandas as pd

# Initialisation des données
@st.cache_data
def load_data():
    return pd.DataFrame({
        "Nom du véhicule": ["Tesla Model 3", "Nissan Leaf", "Renault Zoe", "Hyundai Kona Electric"],
        "Type": ["Vente", "Location", "Location", "Vente"],
        "Prix (CFA/jour ou total)": [40000, 50, 45, 37000],
        "Autonomie (km)": [500, 270, 395, 470],
        "Disponibilité": ["Disponible", "Non disponible", "Disponible", "Disponible"]
    })

# Charger les données
df = load_data()


# Titre de l'application
st.title("Gestion des Ventes et Locations de Véhicules Électriques")

# Menu de navigation
menu = st.sidebar.radio("Menu", ["Accueil", "Liste des véhicules", "Ajouter un véhicule","Vehicule de remplacement","Mise à jour"])

if menu == "Accueil":
    st.header("DALAL JAAM")
    st.write("""
    Cette plateforme permet de gérer efficacement les ventes et locations de véhicules électriques. 
    Utilisez le menu pour consulter la liste des véhicules, en ajouter ou mettre à jour leur état.
    """)

elif menu == "Liste des véhicules":
    st.header("Liste des véhicules")
    # Filtrage des véhicules
    type_vehicule = st.selectbox("Filtrer par type", ["Tous", "Vente", "Location"])
    disponibilite = st.checkbox("Afficher uniquement les véhicules disponibles", False)
    
    # Appliquer les filtres
    filtered_df = df.copy()
    if type_vehicule != "Tous":
        filtered_df = filtered_df[filtered_df["Type"] == type_vehicule]
    if disponibilite:
        filtered_df = filtered_df[filtered_df["Disponibilité"] == "Disponible"]
    
    st.write(f"{len(filtered_df)} véhicule(s) trouvé(s).")
    st.dataframe(filtered_df)

elif menu == "Ajouter un véhicule":
    st.header("Ajouter un nouveau véhicule")
    with st.form("Ajouter un véhicule"):
        nom = st.text_input("Nom du véhicule")
        type_vehicule = st.selectbox("Type", ["Vente", "Location"])
        prix = st.number_input("Prix (€/jour ou total)", min_value=0.0, step=0.1)
        autonomie = st.number_input("Autonomie (km)", min_value=0, step=10)
        disponibilite = st.selectbox("Disponibilité", ["Disponible", "Non disponible"])
        submit = st.form_submit_button("Ajouter")
        
        if submit:
            new_vehicle = {
                "Nom du véhicule": nom,
                "Type": type_vehicule,
                "Prix (CFA/jour ou total)": prix,
                "Autonomie (km)": autonomie,
                "Disponibilité": disponibilite
            }
            df = pd.concat([df, pd.DataFrame([new_vehicle])], ignore_index=True)
            st.success("Véhicule ajouté avec succès !")
            st.dataframe(df)

elif menu == "Mise à jour":
    st.header("Mise à jour de la disponibilité")
    # Sélectionner un véhicule
    vehicule_a_mettre_a_jour = st.selectbox("Sélectionnez un véhicule à mettre à jour", df["Nom du véhicule"])
    if vehicule_a_mettre_a_jour:
        # Trouver la ligne correspondante
        index = df[df["Nom du véhicule"] == vehicule_a_mettre_a_jour].index[0]
        nouvelle_disponibilite = st.selectbox(
            "Nouvelle disponibilité",
            ["Disponible", "Non disponible"],
            index=0 if df.loc[index, "Disponibilité"] == "Disponible" else 1
        )
        if st.button("Mettre à jour"):
            df.at[index, "Disponibilité"] = nouvelle_disponibilite
            st.success(f"Disponibilité du véhicule '{vehicule_a_mettre_a_jour}' mise à jour avec succès !")
            st.dataframe(df)

 ################################
#VISUALIZATIONS 
################################ 
import matplotlib.pyplot as plt

# Visualisation des performances
plt.figure(figsize=(14, 6))

# Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision Entraînement', color='blue')
plt.plot(history.history['val_accuracy'], label='Précision Validation', color='orange')
plt.title('Précision au fil des époques')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()

# Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte Entraînement', color='blue')
plt.plot(history.history['val_loss'], label='Perte Validation', color='orange')
plt.title('Perte au fil des époques')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.show()


            
