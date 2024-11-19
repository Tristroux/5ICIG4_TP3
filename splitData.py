import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Chemin du dataset
base_dir = "RealDataSet"  # Remplacez par le chemin du dossier contenant "bike" et "car"
new_dir = os.path.join(base_dir, "New")  # Dossier contenant les images redimensionnées

# Structure des sous-dossiers pour l'entraînement et le test
train_dir = os.path.join(new_dir, "train")
test_dir = os.path.join(new_dir, "test")

# Créer les dossiers de destination pour les images et labels
def create_train_test_dirs(base_dir, categories):
    """Créer les sous-dossiers pour l'entraînement et le test."""
    for category in categories:
        # Dossiers pour images
        os.makedirs(os.path.join(train_dir, category, "images"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category, "images"), exist_ok=True)
        # Dossiers pour labels
        os.makedirs(os.path.join(train_dir, category, "labels"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category, "labels"), exist_ok=True)

def split_data_and_count(images_dir, categories, test_size=0.2):
    """Séparer les données en ensemble d'entraînement et de test, et compter les images et labels."""
    for category in categories:
        category_img_dir = os.path.join(images_dir, category, "images")
        category_label_dir = os.path.join(images_dir, category, "labels")
        
        # Obtenir toutes les images et labels
        image_files = [f for f in os.listdir(category_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Séparer les données en ensembles d'entraînement et de test
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
        
        # Déplacer les images et labels dans les dossiers appropriés
        for train_file in train_files:
            # Déplacer image
            shutil.move(os.path.join(category_img_dir, train_file), os.path.join(train_dir, category, "images", train_file))
            # Déplacer label
            label_file = train_file.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
            shutil.move(os.path.join(category_label_dir, label_file), os.path.join(train_dir, category, "labels", label_file))
        
        for test_file in test_files:
            # Déplacer image
            shutil.move(os.path.join(category_img_dir, test_file), os.path.join(test_dir, category, "images", test_file))
            # Déplacer label
            label_file = test_file.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
            shutil.move(os.path.join(category_label_dir, label_file), os.path.join(test_dir, category, "labels", label_file))
        
        # Compter le nombre d'images dans chaque groupe
        num_train_images = len(train_files)
        num_test_images = len(test_files)
        
        print(f"{category.capitalize()} - Entraînement : {num_train_images} images, Test : {num_test_images} images")

# Créer les dossiers de destination pour les données d'entraînement et de test
create_train_test_dirs(new_dir, ["bike", "car"])

# Séparer les données et compter les images et labels
split_data_and_count(new_dir, ["bike", "car"])
