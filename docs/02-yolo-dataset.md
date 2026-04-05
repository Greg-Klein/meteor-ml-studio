# 2. Comprendre un dataset YOLO 🗂️

## Structure generale

Un dataset YOLO ressemble souvent a ceci :

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

## 📁 Le role de chaque dossier

### `images/`

Contient les images brutes.

### `labels/`

Contient les fichiers d'annotation associes aux images.

### `dataset.yaml`

Contient la configuration du dataset :

- ou se trouvent les images
- ou se trouvent les labels
- quelles sont les classes

## 🔗 Correspondance image / label

Chaque image doit avoir un fichier `.txt` du meme nom.

Exemple :

- `images/train/frame_001.jpg`
- `labels/train/frame_001.txt`

YOLO fait la correspondance grace au nom.

## 🏷️ Contenu d'un fichier label

Exemple :

```txt
0 0.512 0.438 0.120 0.045
```

Cela veut dire :

- `0` : identifiant de classe
- `0.512` : centre X
- `0.438` : centre Y
- `0.120` : largeur
- `0.045` : hauteur

Les coordonnees sont normalisees entre `0` et `1`.

## Image positive

Une image positive contient un meteore.
Son fichier label contient donc une ligne avec la boite du meteore.

## Image negative

Une image negative ne contient pas de meteore.
Son fichier label est vide.

Cela reste utile :

- le modele apprend aussi ce qu'il ne faut pas detecter

## Exemple de `dataset.yaml`

```yaml
path: /chemin/vers/dataset
train: images/train
val: images/val

names:
  0: meteor
```

## ✅ A retenir

Un dataset YOLO, c'est simplement :

- des images
- des labels
- un fichier `dataset.yaml`
