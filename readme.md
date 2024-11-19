Introduction : 
1. Dans les images du TP précédent on avait une grosse incohérence en arriere plan. La plupart des images de moto d'entrainement n'ont pas d'arriere plan, donc elles sont sur fond plan. Alors que celles des voitures ont des arrieres plans, routes, vilages, terre...
Par contre dans les images de tests c'est l'inverse. Les motos ont un arriere plan et les voiture non. Donc le modèle s'est entrainé de facon à associé plutot les motos avec des images qui ont un fond blanc. Du moins pour lui les motos ont un fond blanc. Et pour lui les voitures n'ont pas de fond blanc.

2. J'ai donc repris toutes les images et je les ai mélangés dans test et train afin d'avoir pour les deux objets, des images avec fond blanc et sans dans train et aussi dans test. (je n'ai pas regardé si la répartition en blanc ou non était égale).

3. Je n'avais pas fait le tp2 donc dans un soucis de temps j'ai demandé a chatGPT de me générer un entrainement de SVM sur mon nouveau DATASET. (il y avait peut etre des spécificité dans le TP2 qui ne sont donc pas reprises ici).

Données en entrée du CNN
1. c'est le programme FectDataSet.py