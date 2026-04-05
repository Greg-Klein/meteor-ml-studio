from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def count_labels(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.suffix.lower() == ".txt")


def inspect_dataset(dataset_root: Path) -> int:
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    dataset_yaml = dataset_root / "dataset.yaml"

    print(f"Dataset root: {dataset_root}")
    print(f"images/train: {count_images(images_train)}")
    print(f"images/val: {count_images(images_val)}")
    print(f"labels/train: {count_labels(labels_train)}")
    print(f"labels/val: {count_labels(labels_val)}")
    print(f"dataset.yaml existe: {'oui' if dataset_yaml.exists() else 'non'}")
    return 0


def write_yaml(dataset_root: Path, class_name: str) -> int:
    dataset_root.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    print(f"Fichier ecrit: {yaml_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Outils dataset YOLO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("dataset_root", type=Path)

    write_yaml_parser = subparsers.add_parser("write-yaml")
    write_yaml_parser.add_argument("dataset_root", type=Path)
    write_yaml_parser.add_argument("class_name")

    args = parser.parse_args()

    if args.command == "inspect":
        return inspect_dataset(args.dataset_root.expanduser())
    if args.command == "write-yaml":
        return write_yaml(args.dataset_root.expanduser(), args.class_name)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
