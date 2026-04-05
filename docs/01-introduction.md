# 1. Introduction 👋

## L'idee de base

Un modele de vision comme YOLO est un programme qui apprend a reconnaitre des objets dans une image.

Dans ton cas, l'objet que l'on veut reconnaitre est :

- `meteor`

Le modele ne comprend pas ce qu'est un meteore par magie.
Il apprend a partir d'exemples.

## 📥 Ce qu'on donne au modele

Pour apprendre, il lui faut :

- des images
- pour chaque image positive, une annotation qui dit ou se trouve le meteore

Ces annotations prennent la forme de boites englobantes.

## 📤 Ce que le modele rend ensuite

Quand le modele a ete entraine, on peut lui donner une nouvelle image.
Il va alors essayer de repondre a la question :

- "Est-ce que je vois un meteore ?"
- "Si oui, ou est-il ?"

Il renvoie alors :

- une boite
- une classe
- un score de confiance

## Pourquoi YOLO

YOLO est populaire parce qu'il est :

- relativement simple a utiliser
- rapide
- tres bien documente
- adapte a la detection d'objet

Pour un premier projet, c'est un tres bon choix.

## ✅ Ce qu'il faut retenir

YOLO ne remplace pas le dataset.
Sans bon dataset, il n'y a pas de bon modele.

Le vrai coeur du projet est donc :

- la collecte des images
- la qualite des annotations
- la bonne separation entre apprentissage et evaluation
