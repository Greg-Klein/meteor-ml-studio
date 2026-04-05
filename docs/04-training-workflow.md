# 4. Workflow d'entrainement 🚀

## Vue d'ensemble

Un premier entrainement reussi ne veut pas dire "modele parfait".

Cela veut simplement dire :

- le dataset est bien structure
- l'entrainement se lance
- le modele apprend quelque chose
- on peut ensuite ameliorer progressivement

## Etape 1 : collecter 📸

On rassemble :

- des images avec meteore
- des images sans meteore

Les negatives sont tres importantes, car elles apprennent au modele a eviter les faux positifs.

## Etape 2 : annoter ✍️

Chaque meteore doit avoir une boite englobante.

Regles simples au debut :

- boite assez serree
- pas besoin d'etre parfaite au pixel pres
- etre coherent est plus important qu'etre ultra precis

## Etape 3 : preparer le dataset 🗂️

On cree :

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`
- `dataset.yaml`

## Etape 4 : lancer l'entrainement 🏋️

Exemple typique :

```bash
yolo detect train data=dataset.yaml model=yolo11n.pt epochs=50 imgsz=640
```

Ici :

- `data=dataset.yaml` indique le dataset
- `model=yolo11n.pt` indique le modele de depart
- `epochs=50` indique le nombre de passes d'entrainement
- `imgsz=640` indique la taille des images redimensionnees

## Etape 5 : lire les resultats 📈

YOLO produit en general :

- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `weights/best.pt`
- `weights/last.pt`

Le plus important au debut :

- `best.pt` = meilleur modele
- `results.png` = evolution de l'entrainement

## Etape 6 : tester 🔬

On prend le modele `best.pt` et on fait des predictions sur de nouvelles images.

But :

- voir s'il detecte bien les meteores
- observer les faux positifs
- observer les faux negatifs

## Etape 7 : iterer 🔁

L'apprentissage machine est iteratif.

On recommence en ameliorant :

- le dataset
- les annotations
- les cas difficiles

## ✅ A retenir

Le premier entrainement sert surtout a valider la chaine complete.
La performance vient ensuite avec les iterations.
