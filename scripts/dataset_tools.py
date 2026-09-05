from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
T = TypeVar("T")


def whole_number(value: str) -> int:
    number = float(value)
    if not number.is_integer():
        raise argparse.ArgumentTypeError(f"Nombre entier attendu: {value}")
    return int(number)


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "_meta" not in path.parts
        and not path.stem.endswith("_annotated")
    )


def list_labels(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".txt")


def inspect_dataset(dataset_root: Path) -> int:
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    dataset_yaml = dataset_root / "dataset.yaml"

    train_images = list_images(images_train)
    val_images = list_images(images_val)
    train_labels = list_labels(labels_train)
    val_labels = list_labels(labels_val)
    all_labels = train_labels + val_labels
    positive_labels = sum(bool(path.read_text(encoding="utf-8").strip()) for path in all_labels)
    negative_labels = len(all_labels) - positive_labels

    print(f"Dataset root: {dataset_root}")
    print(f"images/train: {len(train_images)}")
    print(f"images/val: {len(val_images)}")
    print(f"images/total: {len(train_images) + len(val_images)}")
    print(f"labels/train: {len(train_labels)}")
    print(f"labels/val: {len(val_labels)}")
    print(f"positives annotees: {positive_labels}")
    print(f"faux positifs: {negative_labels}")
    print(f"dataset.yaml existe: {'oui' if dataset_yaml.exists() else 'non'}")
    return 0


def write_yaml(
    dataset_root: Path,
    class_name: str,
    declared_root: Path | None = None,
) -> int:
    dataset_root.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str((declared_root or dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    yaml_path = dataset_root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    print(f"Fichier ecrit: {yaml_path}")
    return 0


def read_positive_annotations(manifest_path: Path) -> list[tuple[Path, Path]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifeste d'annotations introuvable: {manifest_path}")

    annotations: list[tuple[Path, Path]] = []
    seen_names: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"image", "label"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError("Le manifeste doit contenir les colonnes 'image' et 'label'.")

        for row in reader:
            image_path = Path(row["image"]).expanduser()
            label_path = Path(row["label"]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Image positive introuvable: {image_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"Label positif introuvable: {label_path}")
            if not label_path.read_text(encoding="utf-8").strip():
                raise ValueError(f"Label positif vide: {label_path}")
            if image_path.stem in seen_names:
                raise ValueError(f"Image positive dupliquee: {image_path.stem}")
            seen_names.add(image_path.stem)
            annotations.append((image_path, label_path))

    if not annotations:
        raise ValueError("Le manifeste ne contient aucune annotation positive.")
    return sorted(annotations, key=lambda item: str(item[0]))


def sample_negatives(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or limit >= len(paths):
        return paths

    rng = random.Random(seed)
    by_night: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_night[path.parent.name].append(path)

    queues: dict[str, deque[Path]] = {}
    nights = sorted(by_night)
    rng.shuffle(nights)
    for night, night_paths in by_night.items():
        rng.shuffle(night_paths)
        queues[night] = deque(night_paths)

    selected: list[Path] = []
    while len(selected) < limit:
        added = False
        for night in nights:
            if queues[night] and len(selected) < limit:
                selected.append(queues[night].popleft())
                added = True
        if not added:
            break
    return sorted(selected)


def split_items(items: list[T], val_percent: float, seed: int) -> tuple[list[T], list[T]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_percent / 100))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def negative_output_name(path: Path) -> str:
    return f"negative-{path.parent.name}-{path.name}"


def populate_split(
    staging_root: Path,
    split: str,
    positives: list[tuple[Path, Path]],
    negatives: list[Path],
    manifest_rows: list[dict[str, str]],
) -> None:
    images_dir = staging_root / "images" / split
    labels_dir = staging_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, source_label in positives:
        image_name = f"positive-{image_path.name}"
        label_name = f"positive-{image_path.stem}.txt"
        destination_image = images_dir / image_name
        destination_label = labels_dir / label_name
        link_or_copy(image_path, destination_image)
        shutil.copy2(source_label, destination_label)
        manifest_rows.append({
            "split": split,
            "kind": "positive",
            "source": str(image_path),
            "image": str(destination_image.relative_to(staging_root)),
            "label": str(destination_label.relative_to(staging_root)),
        })

    for image_path in negatives:
        image_name = negative_output_name(image_path)
        label_name = f"{Path(image_name).stem}.txt"
        destination_image = images_dir / image_name
        destination_label = labels_dir / label_name
        link_or_copy(image_path, destination_image)
        destination_label.write_text("", encoding="ascii")
        manifest_rows.append({
            "split": split,
            "kind": "negative",
            "source": str(image_path),
            "image": str(destination_image.relative_to(staging_root)),
            "label": str(destination_label.relative_to(staging_root)),
        })


def activate_dataset(staging_root: Path, dataset_root: Path) -> Path | None:
    backup_path: Path | None = None
    if dataset_root.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = dataset_root.with_name(f"{dataset_root.name}-backup-{timestamp}")
        dataset_root.rename(backup_path)

    try:
        staging_root.rename(dataset_root)
    except Exception:
        if backup_path is not None and not dataset_root.exists():
            backup_path.rename(dataset_root)
        raise
    return backup_path


def build_dataset(
    dataset_root: Path,
    annotations_manifest: Path,
    negatives_dir: Path,
    max_negatives: int,
    val_percent: float,
    seed: int,
    class_name: str,
) -> int:
    if not 1 <= val_percent <= 50:
        raise ValueError("Le pourcentage de validation doit etre compris entre 1 et 50.")
    if max_negatives < 0:
        raise ValueError("Le nombre maximal de faux positifs ne peut pas etre negatif.")

    dataset_root = dataset_root.expanduser().resolve()
    annotations_manifest = annotations_manifest.expanduser().resolve()
    negatives_dir = negatives_dir.expanduser().resolve()
    positives = read_positive_annotations(annotations_manifest)
    all_negatives = list_images(negatives_dir)
    if not all_negatives:
        raise ValueError(f"Aucun faux positif trouve dans: {negatives_dir}")
    negatives = sample_negatives(all_negatives, max_negatives, seed)

    train_positives, val_positives = split_items(positives, val_percent, seed)
    train_negatives, val_negatives = split_items(negatives, val_percent, seed + 1)

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = dataset_root.parent / f".{dataset_root.name}-build-{uuid.uuid4().hex}"
    manifest_rows: list[dict[str, str]] = []
    try:
        populate_split(staging_root, "train", train_positives, train_negatives, manifest_rows)
        populate_split(staging_root, "val", val_positives, val_negatives, manifest_rows)
        write_yaml(staging_root, class_name, declared_root=dataset_root)
        with (staging_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("split", "kind", "source", "image", "label"),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        backup_path = activate_dataset(staging_root, dataset_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    print("Dataset construit avec succes.")
    print(f"Positives: {len(positives)} ({len(train_positives)} train, {len(val_positives)} val)")
    print(f"Faux positifs: {len(negatives)} ({len(train_negatives)} train, {len(val_negatives)} val)")
    print(f"Images totales: {len(positives) + len(negatives)}")
    print(f"Dataset actif: {dataset_root}")
    if backup_path is not None:
        print(f"Ancien dataset sauvegarde: {backup_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Outils dataset YOLO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("dataset_root", type=Path)

    write_yaml_parser = subparsers.add_parser("write-yaml")
    write_yaml_parser.add_argument("dataset_root", type=Path)
    write_yaml_parser.add_argument("class_name")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("dataset_root", type=Path)
    build_parser.add_argument("annotations_manifest", type=Path)
    build_parser.add_argument("negatives_dir", type=Path)
    build_parser.add_argument("--max-negatives", type=whole_number, default=600)
    build_parser.add_argument("--val-percent", type=float, default=20)
    build_parser.add_argument("--seed", type=whole_number, default=42)
    build_parser.add_argument("--class-name", default="meteor")

    args = parser.parse_args()

    if args.command == "inspect":
        return inspect_dataset(args.dataset_root.expanduser())
    if args.command == "write-yaml":
        return write_yaml(args.dataset_root.expanduser(), args.class_name)
    if args.command == "build":
        return build_dataset(
            dataset_root=args.dataset_root,
            annotations_manifest=args.annotations_manifest,
            negatives_dir=args.negatives_dir,
            max_negatives=args.max_negatives,
            val_percent=args.val_percent,
            seed=args.seed,
            class_name=args.class_name,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
