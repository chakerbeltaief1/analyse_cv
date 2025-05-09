import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# === Chemin vers le dataset nettoyé ===
file_path = r'C:\analyse_cv\Backend\data\cleaned_resume\cleaned_resume.csv'
model_directory = r'C:\analyse_cv\Backend\data\model'

# === S'assurer que le dossier 'model' existe ===
os.makedirs(model_directory, exist_ok=True)

# === Chargement du dataset ===
df = pd.read_csv(file_path)

# === Nettoyage préliminaire (si besoin) ===
df.dropna(subset=['Resume_Details', 'Category'], inplace=True)
df['Resume_Details'] = df['Resume_Details'].str.lower()

# === Séparer les données ===
X = df['Resume_Details']  # Texte des CV
y = df['Category']        # Catégories cibles

# === Vectorisation TF-IDF ===
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# === Séparation entraînement / test ===
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# === Modèle Random Forest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# === Prédictions et évaluation ===
y_pred = rf_model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# === Sauvegarde du modèle et du vecteur ===
model_path = os.path.join(model_directory, 'random_forest_model.pkl')
vectorizer_path = os.path.join(model_directory, 'tfidf_vectorizer.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print(f"\n💾 Modèle et vecteur TF-IDF sauvegardés dans : {model_directory}")
