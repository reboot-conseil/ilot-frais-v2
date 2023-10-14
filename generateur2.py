import os
import numpy as np
from PIL import Image, ImageDraw
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

# Fonction pour générer une image avec des îlots de chaleur ou une zone normale
def generate_image():
    # Taille de l'image
    width, height = 256, 256

    # On génère un nombre aléatoire pour décider si l'image sera un îlot de chaleur ou une image normale
    is_heat_island = random.choice([True, False])

    # Nouvelle image vide avec un fond noir pour chaque image
    image = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(image)

    # Génération des centres de chaleur dispersés aléatoirement pour les îlots de chaleur
    num_heat_centers = random.randint(5, 15)
    heat_centers = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_heat_centers)]

    # Génération des variations de température pour chaque pixel
    for x in range(width):
        for y in range(height):
            # On calcule la température initiale du pixel en fonction de sa distance aux centres de chaleur
            temperatures_heat = [random.uniform(0, 1) for _ in range(num_heat_centers)]
            for i, (center_x, center_y) in enumerate(heat_centers):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                temperatures_heat[i] *= max(0, 1 - distance / 50)  # Ajuster le 50 pour contrôler la portée de chaque centre de chaleur

            # On calcule la température finale du pixel en combinant les températures des îlots de chaleur
            temperature_heat = sum(temperatures_heat)

            # On génére une température aléatoire pour la zone normale
            temperature_normal = random.uniform(0, 1)

            # On combine la température de l'îlot de chaleur et de la zone normale pour chaque pixel
            if is_heat_island:
                temperature = temperature_heat
                label = 1  # 1 pour les îlots de chaleur
            else:
                temperature = (temperature_heat + temperature_normal) / 2
                label = -1  # -1 pour les zones normales

            # On définit la couleur en utilisant une palette de couleurs spécifique
            if temperature <= 0.2:
                color = (0, 0, 255)  # Bleu (froid)
            elif temperature <= 0.4:
                color = (0, 128, 0)  # Vert (tiède)
            elif temperature <= 0.6:
                color = (255, 255, 0)  # Jaune (un peu chaud)
            elif temperature <= 0.8:
                color = (255, 165, 0)  # Orange (chaud)
            else:
                color = (255, 0, 0)  # Rouge (très chaud)

            # On dessine le pixel avec la couleur calculée
            draw.point((x, y), fill=color)

    return image, label

# Nombre d'images à générer
num_images = 100

# Tableaux pour stocker les images et leurs étiquettes
images = []
labels = []

for _ in range(num_images):
    image, label = generate_image()
    images.append(image)
    labels.append(label)

# Affichage des images en itérant dans le tableau
# for i, image in enumerate(images):
#     plt.figure()
#     plt.imshow(image)
#     if labels[i] == 1:
#         plt.title(f'Image {i + 1}: Ilot de Chaleur')
#     else:
#         plt.title(f'Image {i + 1}: Zone Normale')
#     plt.axis('off')

# On convertit les images en tableaux NumPy aplaties
image_vectors = []
for image in images:
    image_vector = np.array(image).reshape(-1)
    image_vectors.append(image_vector)

# On convertit la liste d'images aplaties en un tableau NumPy
X = np.array(image_vectors) # Tableau avec chaque ligne correspondante à une image [10 x 65536]
y = np.array(labels) # Tableau avec les etiquettes correspondante aux 10 images

# On divise les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Liste des valeurs de C à tester
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Liste pour stocker les précisions du modèle pour chaque valeur de C
accuracies = []

# Entraînement du modèle et calcul de la précision pour chaque valeur de C
for C in C_values:
    # Créez un classificateur SVM linéaire avec la valeur de C actuelle
    clf = svm.SVC(kernel='linear', C=C)

    # Entraînez le modèle
    clf.fit(X_train, y_train)

    # Prédisez les étiquettes sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Mesurez la précision du modèle
    accuracy = accuracy_score(y_test, y_pred)

    # Ajoutez la précision à la liste des précisions
    accuracies.append(accuracy)

# Affichez le graphique d'erreur en fonction de la valeur de C
plt.figure()
plt.plot(C_values, accuracies, marker='o')
plt.title('Précision du modèle SVM en fonction de la valeur de C')
plt.xlabel('Valeur de C')
plt.ylabel('Précision')
plt.xscale('log')  # Échelle logarithmique pour C
plt.grid(True)
plt.show()