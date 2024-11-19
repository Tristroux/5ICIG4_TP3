import os
import cv2

# Chemin du dataset
base_dir = "RealDataSet"  # Remplacez par le chemin du dossier contenant "bike" et "car"

# Chemins pour les dossiers originaux et modifiés
old_dir = os.path.join(base_dir, "Old")
new_dir = os.path.join(base_dir, "New")

# Structure finale des sous-dossiers
categories = {
    "bike": [0, 1],  # Label pour bike
    "car": [1, 0],   # Label pour car
}

# Taille d'image souhaitée
img_size = (64, 64)

def create_directory_structure(base_path, categories):
    """Créer les dossiers nécessaires pour l'arborescence finale."""
    for category in categories:
        # Chemins pour les nouvelles images et labels
        os.makedirs(os.path.join(base_path, category, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_path, category, "labels"), exist_ok=True)

def process_images(old_dir, new_dir, categories, img_size):
    """Traiter les images des dossiers d'origine."""
    for category, label in categories.items():
        input_path = os.path.join(old_dir, category)  # Dossier d'origine
        output_img_path = os.path.join(new_dir, category, "images")  # Dossier des images
        output_label_path = os.path.join(new_dir, category, "labels")  # Dossier des labels
        
        # Compteur pour le renommage
        counter = 1
        
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ignorer les fichiers non-image
                continue
            
            try:
                # Lire et redimensionner l'image
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img_resized = cv2.resize(img, img_size)
                
                # Nouveau nom de fichier
                new_file_name = f"{category}_{counter}.jpg"
                new_file_path = os.path.join(output_img_path, new_file_name)
                
                # Sauvegarder l'image redimensionnée
                cv2.imwrite(new_file_path, img_resized)
                
                # Création du fichier label avec le même nom
                label_file_name = f"{category}_{counter}.txt"
                label_file_path = os.path.join(output_label_path, label_file_name)
                
                # Sauvegarder le label dans le fichier correspondant
                with open(label_file_path, "w") as lf:
                    lf.write(f"{label[0]},{label[1]}")
                
                counter += 1
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {file_path}: {e}")


# Création de l'arborescence des dossiers
create_directory_structure(new_dir, categories)

# Traitement des images
process_images(old_dir, new_dir, categories, img_size)

print("Traitement terminé.")
