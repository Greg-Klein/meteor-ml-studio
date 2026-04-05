# 6. Glossaire 📖

## Annotation

Information associee a une image pour indiquer au modele ce qu'il doit apprendre.

## Batch size

Nombre d'images traitees en meme temps pendant l'entrainement.

## Classe

Categorie d'objet a detecter.
Dans ton cas :

- `meteor`

## Confidence

Score de confiance associe a une detection.
Plus il est eleve, plus le modele est sur de lui.

## Dataset

Ensemble d'images et de labels utilises pour entrainer et evaluer un modele.

## Epoch

Une passe complete sur l'ensemble `train`.

## Inference

Le fait d'utiliser un modele entraine pour faire une prediction sur une nouvelle image.

## Label

Fichier ou information qui indique au modele ou se trouve l'objet dans l'image.

## Loss

Mesure de l'erreur du modele pendant l'entrainement.
En general, plus elle baisse, mieux c'est.

## mAP

Metrique classique en detection d'objet.
Elle sert a mesurer la qualite globale des detections.

## Model

Fichier de poids appris par YOLO, par exemple :

- `best.pt`

## Prediction

Resultat produit par le modele sur une image :

- boites
- classes
- scores

## Train

Partie du dataset utilisee pour apprendre.

## Validation (`val`)

Partie du dataset utilisee pour verifier la qualite du modele pendant l'entrainement.

## YOLO

Famille de modeles de detection d'objet, populaire pour sa simplicite et sa rapidite.
