import streamlit as st
import requests
import pandas as pd
import json
import os
import base64
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Analyse et Classification Intelligente de Données à partir de CV",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour charger le logo
def load_logo():
    """
    Charge et retourne le chemin du logo s'il existe, sinon retourne None.
    """
    # Liste des chemins possibles pour le logo
    logo_paths = [
        r"C:\analyse_cv\frontend\logo\logo-unilog.png",
    ]
    
    for path in logo_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    
    return None

# Variables globales
API_URL = "http://localhost:5000"  # URL de l'API Flask

# Fonction pour vérifier l'état de l'API
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unavailable", "error": f"Code d'état: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unavailable", "error": str(e)}

# Fonction pour créer un graphique de probabilités de classification
def create_classification_chart(classification_data):
    if not classification_data:
        return None
    
    # Extraction des probabilités
    probabilities = classification_data.get("all_probabilities", {})
    
    # Création du dataframe pour le graphique
    df = pd.DataFrame({
        "Catégorie": list(probabilities.keys()),
        "Probabilité": list(probabilities.values())
    })
    
    # Tri des données par probabilité
    df = df.sort_values("Probabilité", ascending=False)
    
    # Création du graphique avec Plotly
    fig = px.bar(
        df, 
        x="Catégorie", 
        y="Probabilité",
        text="Probabilité",
        color="Probabilité",
        color_continuous_scale="Viridis",
        title="Probabilités de classification par catégorie"
    )
    
    # Customisation du graphique
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Catégorie de CV",
        yaxis_title="Probabilité",
        yaxis=dict(range=[0, 1]),
        coloraxis_showscale=False
    )
    
    return fig

# Fonction pour créer un graphique radar des compétences
def create_skills_radar(skills: List[str]):
    if not skills or len(skills) == 0:
        return None
    
    # Regroupement des compétences par catégorie
    skill_categories = {
        "Langages de programmation": ["Python", "Java", "C++", "JavaScript", "HTML", "CSS"],
        "Data Science": ["Machine Learning", "Deep Learning", "Data Analysis", "Pandas", "Numpy", "Scikit-learn", "TensorFlow", "Keras", "LLM"],
        "Base de données": ["SQL", "NoSQL"],
        "Frameworks": ["Django", "Flask", "React"],
        "Visualisation": ["Power BI", "Tableau"],
        "Big Data": ["Spark", "Hadoop", "Hive", "Pig"],
        "Autres": ["Excel"]
    }
    
    # Initialisation des scores par catégorie
    category_scores = {cat: 0 for cat in skill_categories.keys()}
    category_counts = {cat: 0 for cat in skill_categories.keys()}
    
    # Calcul des scores
    for skill in skills:
        for category, cat_skills in skill_categories.items():
            for cat_skill in cat_skills:
                if cat_skill.lower() in skill.lower():
                    category_scores[category] += 1
                    category_counts[category] += 1
                    break
    
    # Normalisation des scores (0 à 1)
    max_possible = max(len(cat_skills) for cat_skills in skill_categories.values())
    normalized_scores = {}
    for cat, score in category_scores.items():
        if category_counts[cat] > 0:
            normalized_scores[cat] = min(score / max_possible, 1.0)
        else:
            normalized_scores[cat] = 0.0
    
    # Création du graphique radar avec Plotly
    categories = list(normalized_scores.keys())
    values = list(normalized_scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Compétences'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Profil de compétences"
    )
    
    return fig

# Fonction pour afficher les expériences professionnelles
def display_experiences(experiences: List[Dict[str, Any]]):
    if not experiences or len(experiences) == 0:
        st.info("Aucune expérience professionnelle détectée")
        return
    
    # Tri des expériences par date (la plus récente d'abord)
    sorted_exp = sorted(
        experiences, 
        key=lambda x: x.get("Date Début", "0000"), 
        reverse=True
    )
    
    for i, exp in enumerate(sorted_exp):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader(f"{exp.get('Date Début', 'N/A')} - {exp.get('Date Fin', 'N/A')}")
            st.caption(f"Type: {exp.get('Type', 'N/A')}")
        
        with col2:
            st.subheader(exp.get("Société", "Entreprise non spécifiée"))
            
            if exp.get("sujet"):
                st.markdown(f"**Sujet:** {exp.get('sujet')}")
            
            if exp.get("Technologies") and len(exp.get("Technologies")) > 0:
                st.markdown("**Technologies utilisées:**")
                st.write(", ".join(exp.get("Technologies")))
            
            if exp.get("Responsabilités") and len(exp.get("Responsabilités")) > 0:
                st.markdown("**Responsabilités:**")
                for resp in exp.get("Responsabilités"):
                    st.markdown(f"- {resp}")
        
        if i < len(sorted_exp) - 1:
            st.markdown("---")

# Fonction pour afficher la formation
def display_education(education: List[Dict[str, Any]]):
    if not education or len(education) == 0:
        st.info("Aucune formation détectée")
        return
    
    # Tri des formations par date (la plus récente d'abord)
    sorted_edu = sorted(
        education, 
        key=lambda x: x.get("Date Début", "0000"), 
        reverse=True
    )
    
    for i, edu in enumerate(sorted_edu):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader(f"{edu.get('Date Début', 'N/A')} - {edu.get('Date Fin', 'N/A')}")
            st.caption(f"Type: {edu.get('Type', 'N/A')}")
        
        with col2:
            st.subheader(edu.get("Diplôme", "Diplôme non spécifié"))
            if edu.get("Institution"):
                st.markdown(f"**Institution:** {edu.get('Institution')}")
        
        if i < len(sorted_edu) - 1:
            st.markdown("---")

# Fonction pour afficher les projets
def display_projects(projects: List[Dict[str, Any]]):
    if not projects or len(projects) == 0:
        st.info("Aucun projet détecté")
        return
    
    for i, project in enumerate(projects):
        with st.expander(f"{project.get('title', 'Projet')} {project.get('number', '')}"):
            st.write(project.get('description', 'Pas de description disponible'))
            
            if project.get('technologies') and len(project.get('technologies')) > 0:
                st.markdown("**Technologies utilisées:**")
                st.write(", ".join(project.get('technologies')))
        
        if i < len(projects) - 1:
            st.markdown("---")

# Fonction pour analyser un CV via l'API
def analyze_cv(file_bytes):
    try:
        files = {"file": ("cv.pdf", file_bytes, "application/pdf")}
        response = requests.post(f"{API_URL}/analyze", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get("error", str(response.status_code))
            st.error(f"Erreur lors de l'analyse du CV: {error_msg}")
            return None
            
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

# Interface principale
def main():
    # Charger le logo
    logo_path = load_logo()
    
    # Sidebar avec informations sur l'API
    with st.sidebar:
        # Afficher le logo dans la sidebar
        if logo_path:
            try:
                logo_image = Image.open(logo_path)
                st.image(logo_image, width=150)
            except Exception as e:
                pass
        
        st.title("Analyse et Classification Intelligente de Données à partir de CV")
        st.markdown("---")
        
        # Vérification de l'état de l'API
        api_status = check_api_health()
        
        if api_status.get("status") == "healthy":
            st.success("✅ API Connectée")
            st.markdown(f"URL: `{API_URL}`")
            
            # Affichage des informations sur les modèles
            if "models_loaded" in api_status and api_status["models_loaded"]:
                st.success("✅ Modèles de classification chargés")
            else:
                st.warning("⚠️ Modèles de classification non disponibles")
            
            if "spacy_models" in api_status:
                spacy = api_status["spacy_models"]
                if spacy.get("fr", False) and spacy.get("en", False):
                    st.success("✅ Modèles spaCy chargés (FR/EN)")
                else:
                    st.warning(f"⚠️ Modèles spaCy incomplets: FR={spacy.get('fr')}, EN={spacy.get('en')}")
        else:
            st.error(f"❌ API Indisponible: {api_status.get('error', 'Erreur inconnue')}")
            st.warning(f"Vérifiez que l'API Flask fonctionne sur {API_URL}")
        
        st.markdown("---")
        st.markdown("### À propos")
        st.info(
    """
    **📊 Application d'Analyse et Classification Automatique de CV**
    
    Cette solution IA permet le traitement intelligent des CV PDF avec extraction des données clés :

    **🔍 Données extraites :**
    - **Identité** : Nom, coordonnées, localisation
    - **Expérience** : Postes, entreprises, périodes, missions
    - **Formation** : Diplômes, établissements, années
    - **Compétences** : Technologies maîtrisées (Python, SQL, etc.)
    - **Projets** : Réalisations concrètes et technologies utilisées
    - **Langues** : Niveaux certifiés ou déclarés

    **✨ Fonctionnalités avancées :**
    - Classification des profils par domaine (Data, Dev, etc.)
    - Cartographie interactive des compétences techniques

    **🤖 Technologies :** 
    - NLP (Traitement Langage Naturel)
    - Machine Learning (Random Forest)
    - API Flask pour le traitement backend
    - Streamlit pour l'interface utilisateur
    
    **👨‍💻 Créé par Chaker Beltaief**
    """
)
    
    # Corps principal de l'application
    st.title("Analyse et Classification Intelligente de Données à partir de CV")
    st.subheader("Téléchargez un CV au format PDF pour l'analyser")
    st.caption("Projet créé par Chaker Beltaief")
    
    # Zone de téléchargement de fichier
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Affichage du fichier téléchargé
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.info(f"Fichier: {uploaded_file.name}")
            st.info(f"Taille: {uploaded_file.size / 1024:.2f} Ko")
            
            # Bouton d'analyse
            if st.button("Analyser le CV", type="primary"):
                with st.spinner("Analyse en cours..."):
                    # Lecture du fichier
                    file_bytes = uploaded_file.getvalue()
                    
                    # Appel de l'API pour analyser le CV
                    result = analyze_cv(file_bytes)
                    
                    if result:
                        # Stockage des résultats dans la session
                        st.session_state.cv_result = result
                        st.success("Analyse terminée avec succès!")
                        # Utiliser la méthode compatible avec les anciennes versions de Streamlit
                        try:
                            st.rerun()  # Pour Streamlit 1.10.0+
                        except AttributeError:
                            st.experimental_rerun()  # Pour les versions antérieures
        
        # Vérification si des résultats sont disponibles dans la session
        if "cv_result" in st.session_state:
            result = st.session_state.cv_result
            features = result.get("features", {})
            classification = result.get("classification", {})
            
            # Affichage des onglets pour organiser l'information
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
               "📊 Classification", "📞 Contact", "📋 Résumé", "👔 Expériences", "🛠️ Projets", "🎓 Formation", "🔧 Compétences", "🌐 Languages"
            ])
            
            # --- ONGLET 1: CLASSIFICATION ---
            with tab1:
                st.header("Classification du CV")
                
                if classification:
                    category = classification.get("category", "Non classifié")
                    probability = classification.get("probability", 0)
                    
                    st.metric("Catégorie de profil", 
                              category, 
                              f"{probability:.2%}")
                    
                    # Graphique des probabilités
                    st.subheader("Répartition des probabilités")
                    class_chart = create_classification_chart(classification)
                    if class_chart:
                        st.plotly_chart(class_chart, use_container_width=True)
                else:
                    st.warning("La classification n'est pas disponible (les modèles ne sont peut-être pas chargés)")
            
            # --- ONGLET 2: CONTACT ---
            with tab2:
                st.header("Profil et Contact")
                
                # En-tête avec infos principales
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Photo de profil du CV (remplace le logo de la société)
                    # Ici on utilise une image placeholder car l'API n'extrait pas encore la photo du CV
                    st.image("https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png", width=150)
                    st.caption("Photo de profil")
                
                with col2:
                    name = features.get("name", "Nom non détecté")
                    st.header(name)
                    st.subheader(features.get("title", "Poste non détecté"))
                
                st.markdown("---")
                
                # Infos de contact
                contact = features.get("contact", {})
                if contact:
                    cols = st.columns(2)
                    
                    with cols[0]:
                        if contact.get("email"):
                            st.markdown(f"📧 **Email:** {contact.get('email')}")
                        if contact.get("phone"):
                            st.markdown(f"📱 **Téléphone:** {contact.get('phone')}")
                        if contact.get("location"):
                            st.markdown(f"📍 **Localisation:** {contact.get('location')}")
                    
                    with cols[1]:
                        if contact.get("linkedin"):
                            st.markdown(f"🔗 **LinkedIn:** [{contact.get('linkedin')}]({contact.get('linkedin')})")
                        if contact.get("github"):
                            st.markdown(f"💻 **GitHub:** [{contact.get('github')}]({contact.get('github')})")
                        if contact.get("portfolio"):
                            st.markdown(f"🌐 **Portfolio:** [{contact.get('portfolio')}]({contact.get('portfolio')})")
                
                    # Liens sociaux
                    st.markdown("---")
                    social_cols = st.columns(3)
                    
                    with social_cols[0]:
                        if contact.get("linkedin"):
                            st.markdown(f"[🔗 LinkedIn]({contact.get('linkedin')})")
                    
                    with social_cols[1]:
                        if contact.get("github"):
                            st.markdown(f"[💻 GitHub]({contact.get('github')})")
                    
                    with social_cols[2]:
                        if contact.get("portfolio"):
                            st.markdown(f"[🌐 Portfolio]({contact.get('portfolio')})")
                else:
                    st.info("Aucune information de contact détectée")
            
            # --- ONGLET 3: RÉSUMÉ ---
            with tab3:
                st.header("Résumé")
                
                # Résumé
                if features.get("summary"):
                    st.write(features.get("summary"))
                else:
                    st.info("Aucun résumé détecté dans le CV")
            
            # --- ONGLET 4: EXPÉRIENCES ---
            with tab4:
                st.header("Expériences professionnelles")
                display_experiences(features.get("experience", []))
            
            # --- ONGLET 5: PROJETS ---
            with tab5:
                st.header("Projets")
                display_projects(features.get("projects", []))
            
            # --- ONGLET 6: FORMATION ---
            with tab6:
                st.header("Formation")
                display_education(features.get("education", []))
                
            # --- ONGLET 7: COMPÉTENCES ---
            with tab7:
                st.header("Compétences")
                
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    skills = features.get("skills", [])
                    if skills and len(skills) > 0:
                        for skill in skills:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("Aucune compétence détectée")
                
                with col2:
                    radar_chart = create_skills_radar(features.get("skills", []))
                    if radar_chart:
                        st.plotly_chart(radar_chart, use_container_width=True)
            
            # --- ONGLET 8: LANGUES ---
            with tab8:
                st.header("Langues")
                if features.get("languages") and len(features.get("languages")) > 0:
                    for language in features.get("languages"):
                        st.markdown(f"- {language}")
                else:
                    st.info("Aucune langue détectée")

    # Ajouter un pied de page avec la mention de l'auteur
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; padding: 30px;'>Application développée par Chaker Beltaief © 2025</div>", unsafe_allow_html=True)

# Exécution de l'application Streamlit
if __name__ == "__main__":
    main()