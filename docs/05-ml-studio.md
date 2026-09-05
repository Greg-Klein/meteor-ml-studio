# 5. Utiliser Meteor ML Studio 🖥️

## But de l'interface

`Meteor ML Studio` est une petite interface web locale qui sert a piloter les scripts Python.

Elle ne remplace pas YOLO.
Elle sert a te donner une interface plus simple.

## Onglet `Dataset` 🗂️

Cet onglet sert a :

- inspecter un dossier dataset
- construire le dataset depuis les annotations positives et les faux positifs
- regenerer uniquement un `dataset.yaml` si necessaire

Pour construire le dataset :

1. Verifie le chemin du manifeste des annotations positives.
2. Verifie le dossier des faux positifs.
3. Garde 600 faux positifs et 20 % de validation pour le premier entrainement.
4. Clique sur `Construire le dataset`.
5. Clique sur `Inspecter le dataset` et controle les totaux affiches.

Le bouton `Regenerer uniquement dataset.yaml` ne copie aucune image. Il sert
seulement a recreer le fichier de configuration YOLO.

Lors d'une construction reussie, l'ancien dataset est conserve dans un dossier
de sauvegarde voisin. Si la construction echoue, le dataset actif reste intact.

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

La case `Faire defiler automatiquement les logs` controle uniquement le journal :

- decochee, ta position dans les logs et dans la page reste stable
- cochee, le journal suit automatiquement les nouvelles lignes

Pendant un run, le bouton `Lancer l'entrainement` devient automatiquement
`Arreter l'entrainement`. L'arret envoie d'abord une interruption douce a YOLO.
Cela lui laisse le temps de fermer le run et de conserver `best.pt`, `last.pt` et
les resultats deja produits. Si le processus ne repond pas, l'application tente
ensuite une terminaison normale, puis force l'arret en dernier recours. Le bouton
revient a son etat initial lorsque le processus est ferme.

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
