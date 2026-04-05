from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Lancer un entrainement YOLO")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="meteor_train")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"dataset.yaml introuvable: {args.data}")

    print(f"Chargement du modele: {args.model}")
    model = YOLO(args.model)

    print("Demarrage de l'entrainement...")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )
    print("Entrainement termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
