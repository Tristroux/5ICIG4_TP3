import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Chemins des dossiers
data_dir = "RealDataSet/Allimages"  # Remplacez par le chemin de votre dataset
categories = ["bike", "car"]  # Les noms des dossiers correspondant aux classes

# Préparation des données
def load_images(data_dir, categories, img_size=(64, 64)):
    X = []  # Images
    y = []  # Labels
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            try:
                # Chargement et redimensionnement de l'image
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Erreur avec le fichier {file_path}: {e}")
    return np.array(X), np.array(y)

# Charger les données
X, y = load_images(data_dir, categories)

# Normaliser les images (conversion des pixels en valeurs [0, 1])
X = X / 255.0

# Aplatir les images pour les rendre compatibles avec un SVM
X_flatten = X.reshape(len(X), -1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Prédire sur les ensembles d'entraînement et de test
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

# Calcul des accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Affichage des résultats
print(f"Accuracy sur l'ensemble d'entraînement : {train_accuracy:.2f}")
print(f"Accuracy sur l'ensemble de test : {test_accuracy:.2f}")
