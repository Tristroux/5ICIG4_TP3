from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Créer un modèle séquentiel
cnn_model = models.Sequential()

cnn_model.add(layers.Conv2D(4, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
cnn_model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(2, activation='softmax'))

# Compiler le modèle avec binary_crossentropy, Adam et accuracy
cnn_model.compile(
    loss='binary_crossentropy',  # Loss pour la classification binaire
    optimizer='adam',            # Optimiseur Adam
    metrics=['accuracy']         # Métrique pour évaluer la précision
)

# Chemin vers vos images d'entraînement
train_dir = './realDataSet/New/train'

# Dossiers des classes (bike, car)
class_names = ['bike', 'car']

# Initialisation des listes pour les images et les étiquettes
x_train = []
y_train = []
for label, class_name in enumerate(class_names):
    # Dossier des images et des labels pour chaque catégorie
    image_folder = os.path.join(train_dir, class_name, "images")
    label_folder = os.path.join(train_dir, class_name, "labels")

    # Lister les images et labels dans les dossiers respectifs
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    # S'assurer que les fichiers d'images et de labels sont dans le bon ordre
    # (L'ordre des fichiers dans le dossier 'images' doit correspondre à l'ordre des fichiers dans 'labels')
    for img_name, lbl_name in zip(image_files, label_files):
        img_path = os.path.join(image_folder, img_name)
        lbl_path = os.path.join(label_folder, lbl_name)

        # Charger l'image et la convertir en tableau numpy
        img = image.load_img(img_path, target_size=(64, 64))  # Redimensionner si nécessaire
        img_array = image.img_to_array(img)
        x_train.append(img_array)

        # Charger le label
        with open(lbl_path, 'r') as lbl_file:
            label_values = lbl_file.read().strip().split()  # Séparer les valeurs de probabilité
            label_values = [float(val) for val in label_values]  # Convertir en float
        y_train.append(label_values)  # Ajouter le vecteur de probabilités

# Convertir les listes en arrays numpy
x_train = np.array(x_train) / 255.0  # Normalisation des images
y_train = np.array(y_train)

# Entraîner le modèle sur 50 époques
history = cnn_model.fit(
    x_train,             # Données d'entraînement
    y_train,             # Étiquettes d'entraînement
    epochs=50,           # Nombre d'époques
    batch_size=32        # Taille du batch (ajustable selon la mémoire)
)

# Tracer la courbe de l'accuracy
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, graphique 1
plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
plt.plot(history.history['loss'], label='Loss', color='red')
plt.title('Accuracy en fonction des époques')
plt.xlabel('Époques')
plt.ylabel('valeur')
plt.legend()

# Afficher les courbes
plt.tight_layout()
plt.show()