from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def predict_image(
    model_path: str,
    image_path: Path,
    conf: float = 0.25,
    image_size: int = 640,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    logs: list[str] = []
    logs.append(f"Chargement du modele: {model_path}")
    model = YOLO(model_path)

    logs.append(f"Prediction sur: {image_path}")
    logs.append(f"Resolution d'analyse: {image_size}")
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=True,
        imgsz=image_size,
    )

    result = results[0]
    detections: list[dict] = []
    for index, box in enumerate(result.boxes, start=1):
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        label = result.names.get(cls_id, str(cls_id))
        detections.append(
            {
                "index": index,
                "class": label,
                "confidence": round(score, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )

    logs.append("")
    logs.append(f"Nombre de detections: {len(detections)}")
    if detections:
        for item in detections:
            logs.append(
                f"{item['index']}. class={item['class']} conf={item['confidence']:.4f} "
                f"box=({item['x1']}, {item['y1']}, {item['x2']}, {item['y2']})"
            )
    else:
        logs.append("Aucune detection.")

    save_dir = Path(result.save_dir)
    annotated_path = save_dir / image_path.name
    return {
        "logs": "\n".join(logs),
        "detections": detections,
        "annotated_image": str(annotated_path) if annotated_path.exists() else None,
        "save_dir": str(save_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediction YOLO sur une image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    prediction = predict_image(args.model, args.image, args.conf, args.imgsz)
    print(prediction["logs"])
    if prediction["annotated_image"]:
        print(f"Image annotee: {prediction['annotated_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
