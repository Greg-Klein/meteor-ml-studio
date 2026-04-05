# Documentation Meteor ML Studio

Cette documentation est ecrite pour un debutant complet.

Le but est de t'aider a comprendre :

- ce qu'est un modele YOLO 🤖
- comment un dataset d'entrainement est structure 🗂️
- a quoi servent `train` et `val` 🧪
- comment se deroule un entrainement 🚀
- comment utiliser `Meteor ML Studio` 🖥️

## 📚 Ordre de lecture conseille

1. [`01-introduction.md`](./01-introduction.md)
2. [`02-yolo-dataset.md`](./02-yolo-dataset.md)
3. [`03-train-vs-val.md`](./03-train-vs-val.md)
4. [`04-training-workflow.md`](./04-training-workflow.md)
5. [`05-ml-studio.md`](./05-ml-studio.md)
6. [`06-glossary.md`](./06-glossary.md)

## 🌠 Idee generale

Dans ce projet, on veut apprendre a un modele a reconnaitre des meteores dans des images.

Pour y arriver, on suit toujours le meme cycle :

1. collecter des images
2. annoter les meteores
3. preparer un dataset YOLO
4. entrainer un modele
5. tester le modele
6. corriger les erreurs
7. recommencer

Tu n'as pas besoin de tout comprendre d'un coup. Le plus important est de voir comment les pieces s'emboitent entre elles.
