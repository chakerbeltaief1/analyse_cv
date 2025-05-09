# 🧠 Analyse et Classification Intelligente de Données à partir de CV

Ce projet a pour objectif de traiter automatiquement des CV au format PDF, d’en extraire les informations clés (entités) grâce à des techniques avancées de NLP (Natural Language Processing), puis de classifier le profil à l’aide du Machine Learning (Random Forest)


## 🔄 Schéma Global du Pipeline

graph TD;
    A[📄 PDF/CV] --> B[🔍 Extraction texte]
    B --> C[🧹 Prétraitement NLP]
    C --> D[🧠 Extraction d'entités NER]
    D --> E[📊 Vectorisation TF-IDF]
    E --> F[🤖 Classification ML(Random Forest)]
    F --> G[📌 Profil classifié + Données structurées]
    G --> H[📊 Interface Streamlit]
    

## 🧩 Détail des Étapes

### 📄 1. PDF CV
- Point de départ du pipeline : le CV peut être un pdf.

### 🔍 2. Extraction de texte
- Utilisation de bibliothèques telles que **PyMuPDF**, **PDFMiner** .
- Extraction du contenu brut (texte) du CV.

### 🧹 3. Nettoyage + NLP
- Traitement linguistique :
  - Suppression des caractères spéciaux
  - Tokenisation
  - Mise en minuscule
  - Suppression des *stopwords*
  - Lemmatisation / Stemmatisation

### 🧠 4. Extraction d'entités (NER)
- Utilisation de modèles NLP avec **spaCy(fr et en)** .
- Extraction automatique de :
  - Nom & Prénom
  - Email, numéro de téléphone
  - linkedin ,github .......
  - localisation
  - education & expériences professionnelles & projets & skills & languages

### 📊 5. Vectorisation TF-IDF
- Transformation du texte nettoyé en vecteurs numériques avec **TF-IDF**.
- Permet de représenter l’importance relative des termes par rapport à l’ensemble des documents.

### 🤖 6. Classification via Machine Learning
- Entraînement d’un modèle **Random Forest Classifier**.
- Objectif : prédire automatiquement le type de profil (ex de dataset :  *Data Engineer*, *web developer*,HR electrical CONSULTANT ...etc.).
- Probabilités associées à chaque catégorie pour une analyse nuancée


### 📌 7. Résultat final
- Génération d’un **profil classifié**
- Liste structurée des **entités extraites** (nom, email, compétences…etc)
## 🛠️ Technologies utilisées

- **Backend** :
  - Python 3.10.0
  - Flask (API Flask)
  - spaCy (NLP)
  - Random Foreset (ML)
  - PyMuPDF/PDFMiner (extraction PDF)
  - python-docx (extraction DOCX)


---

## 💻 Interface Utilisateur avec Streamlit

Un fichier `streamlit_app.py` est inclus pour permettre aux utilisateurs d’interagir avec le modèle via une interface simple et intuitive.

### Fonctionnalités :
- 📤 Upload d’un CV au format PDF
- 🔎 Affichage du texte extrait et des entités détectées
- 🧠 Classification automatique du profil
- 🧾 Résumé structuré prêt à être exporté

### Lancement :
```bash
streamlit run streamlit_app.py
## 🛠️ Technologies utilisées
- **Frontend** :
  - Streamlit
  - Plotly
  - Matplotlib
  - Pandas
## 📦 Installation

### Prérequis
- Python 3.10.0
- pip

### Installation des dépendances

```bash
cd cv-analyzer
# Créer un environnement virtuel
python -m venv venv
# Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les modèles spaCy
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg
```

## 🚀 Utilisation

### Démarrer l'API Backend

```bash
cd backend
python app.py
```

L'API sera accessible à l'adresse `http://localhost:5000`

### Lancer l'interface Streamlit

```bash
cd frontend
streamlit run streamlit_app.py
```

L'interface sera disponible à l'adresse `http://localhost:8501`
 
 👨‍💻  Projet créé par Chaker Beltaief 

