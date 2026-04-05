# 5. Utiliser Meteor ML Studio 🖥️

## But de l'interface

`Meteor ML Studio` est une petite interface web locale qui sert a piloter les scripts Python.

Elle ne remplace pas YOLO.
Elle sert a te donner une interface plus simple.

## Onglet `Dataset` 🗂️

Cet onglet sert a :

- inspecter un dossier dataset
- generer un `dataset.yaml`

Tu peux l'utiliser si tu prepares un dataset YOLO local.

## Onglet `Train` 🚀

Cet onglet sert a lancer un entrainement.

Champs principaux :

- `Chemin vers dataset.yaml`
- `Modele de depart`
- `Epochs`
- `Image size`
- `Batch size`
- `Dossier des runs`
- `Nom du run`

Quand tu cliques sur `Lancer l'entrainement`, l'interface lance le script Python correspondant et affiche les logs.

## Onglet `Runs` 📊

Cet onglet sert a suivre les entrainements deja effectues.

Tu peux y voir :

- le dernier run detecte
- les modeles `best.pt`
- `results.png`
- `confusion_matrix.png`
- des graphiques de loss et de metriques

## Onglet `Predict` 🔍

Cet onglet sert a tester un modele sur une image.

Tu peux :

- choisir un modele
- fournir une image
- lancer la prediction

Puis tu vois :

- les logs
- l'image annotee
- un tableau des detections

## Ce qu'il faut comprendre

L'interface n'est qu'un tableau de bord.

En dessous, ce sont toujours les scripts Python qui font le travail :

- `scripts/dataset_tools.py`
- `scripts/train.py`
- `scripts/predict.py`

## ✅ A retenir

Si quelque chose ne marche pas, il faut toujours penser a ces deux niveaux :

1. l'interface
2. le script Python en dessous
