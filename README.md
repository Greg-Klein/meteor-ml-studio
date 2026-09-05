# Meteor ML Studio

Interface web locale pour piloter un workflow YOLO autour de la detection de meteores.

## Fonctionnalites V1

- inspection d'un dataset YOLO
- construction d'un dataset equilibre depuis les annotations positives et les faux positifs
- generation d'un `dataset.yaml`
- lancement d'un entrainement YOLO
- suivi optionnel des logs et arret propre d'un entrainement en cours
- prediction sur une image avec un modele
- consultation des modeles et des runs produits

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration via `.env`

Tu peux copier le fichier d'exemple :

```bash
cp .env.example .env
```

Variables utiles :

- `METEOR_ML_STUDIO_DEMO_IMAGE` : image chargee par defaut dans l'onglet `Predict`
- `METEOR_ML_STUDIO_DATASET_DIR` : dossier dataset affiche par defaut
- `METEOR_ML_STUDIO_RUNS_DIR` : dossier des runs YOLO affiche par defaut
- `METEOR_ML_STUDIO_ANNOTATIONS_MANIFEST` : manifeste des annotations positives
- `METEOR_ML_STUDIO_NEGATIVES_DIR` : dossier contenant les faux positifs

Si `.env` n'est pas present, l'application utilise des valeurs par defaut locales.

Le projet peut aussi embarquer une image de demonstration dans :

```text
examples/sample.jpg
```

## Lancer l'interface

```bash
source .venv/bin/activate
python app.py
```

Puis ouvrir l'URL locale affichee dans le terminal.

## Structure attendue pour le dataset

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

Le bouton `Construire le dataset` cree cette structure depuis le manifeste des
annotations positives et le dossier des faux positifs. La valeur conseillee pour
ce projet est de 600 faux positifs et 20 % d'images de validation.

La construction est faite dans un dossier temporaire. L'ancien dataset est
deplace dans un dossier `dataset-backup-<date>` uniquement lorsque le nouveau
dataset est complet.

## Notes

- Cette V1 est faite pour etre simple et pedagogique.
- Les actions d'entrainement et de prediction s'appuient sur des scripts Python du dossier `scripts/`.
- Les logs sont affiches dans l'interface pour aider a comprendre ce qui se passe.

## Documentation

Une documentation debutant est disponible dans [`docs/`](./docs) :

- [`docs/README.md`](./docs/README.md) : point d'entree
- [`docs/01-introduction.md`](./docs/01-introduction.md) : les bases
- [`docs/02-yolo-dataset.md`](./docs/02-yolo-dataset.md) : structure d'un dataset YOLO
- [`docs/03-train-vs-val.md`](./docs/03-train-vs-val.md) : comprendre `train` et `val`
- [`docs/04-training-workflow.md`](./docs/04-training-workflow.md) : entrainement pas a pas
- [`docs/05-ml-studio.md`](./docs/05-ml-studio.md) : utiliser cette interface
- [`docs/06-glossary.md`](./docs/06-glossary.md) : glossaire
