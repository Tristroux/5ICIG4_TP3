Introduction : 
1. Dans les images du TP précédent on avait une grosse incohérence en arriere plan. La plupart des images de moto d'entrainement n'ont pas d'arriere plan, donc elles sont sur fond plan. Alors que celles des voitures ont des arrieres plans, routes, vilages, terre...
Par contre dans les images de tests c'est l'inverse. Les motos ont un arriere plan et les voiture non. Donc le modèle s'est entrainé de facon à associé plutot les motos avec des images qui ont un fond blanc. Du moins pour lui les motos ont un fond blanc. Et pour lui les voitures n'ont pas de fond blanc.

2. J'ai donc repris toutes les images et je les ai mélangés dans test et train afin d'avoir pour les deux objets, des images avec fond blanc et sans dans train et aussi dans test. (je n'ai pas regardé si la répartition en blanc ou non était égale).

3. Je n'avais pas fait le tp2 donc dans un soucis de temps j'ai demandé a chatGPT de me générer un entrainement de SVM sur mon nouveau DATASET. (il y avait peut etre des spécificité dans le TP2 qui ne sont donc pas reprises ici).

Données en entrée du CNN (main.py)
1. c'est le programme FectDataSet.py
2. C'est le programe splitData.py, j'ai séparé en 80/20
Bike - Entraînement : 416 images, Test : 104 images
Car - Entraînement : 400 images, Test : 100 images

3. Entrée on a une image et en sortie la classification. Donc la taille en entrée est 64x64x3 et la taille de sortie va etre 1*

Création du réseau de neurone convolutif (CNN)
3. On reduit la taille de motité donc a ce niveau la on aura 32*32*3

5.
a sortie de la couche précédente (la couche de convolution avec 16 filtres) a une taille de (32, 32, 16), car le MaxPooling précédent a réduit la taille de l'image de moitié (de 64x64 à 32x32), mais le nombre de filtres (profondeur) est de 16.
Input shape avant cette couche de MaxPooling : (32, 32, 16)
Output shape après cette couche de MaxPooling : (16, 16, 16)

7.
La sortie de la couche de convolution avec 32 filtres a une taille de (16, 16, 32), car après la précédente couche de MaxPooling, la taille a été réduite de moitié (de 32x32 à 16x16).
Input shape avant cette couche de MaxPooling : (16, 16, 32)
Output shape après cette couche de MaxPooling : (8, 8, 32)

9. Comme on a deux classes et que l'on cheche a determiné les probalité d'appartenance a chaque classe, alors on doit avoir une sortie de taille 2, pour la probalité d'appartance a la classe 1 et a la classe 2.

Entraînement du CNN
1.La binary_crossentropy est la loss adaptée pour la classification binaire, car elle mesure l'écart entre les probabilités prédites (par la fonction d'activation sigmoid) et les étiquettes réelles (0 ou 1). 

