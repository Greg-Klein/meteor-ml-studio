from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from scripts.predict import predict_image


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
load_dotenv(PROJECT_ROOT / ".env")


def resolve_demo_image() -> Path | None:
    configured = os.environ.get("METEOR_ML_STUDIO_DEMO_IMAGE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return candidate

    local_candidates = [
        PROJECT_ROOT / "examples" / "sample.jpg",
        PROJECT_ROOT / "sample.jpg",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate
    return None


def run_script_stream(script_name: str, *args: str):
    command = [
        sys.executable,
        str(SCRIPTS_DIR / script_name),
        *(str(arg) for arg in args if arg is not None and str(arg) != ""),
    ]
    yield "$ " + " ".join(command) + "\n"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        yield "".join(lines)

    return_code = process.wait()
    lines.append(f"\n[exit code: {return_code}]\n")
    yield "".join(lines)


def list_models(runs_root: str) -> str:
    root = Path(runs_root).expanduser()
    if not root.exists():
        return f"Dossier introuvable: {root}"

    models = sorted(root.rglob("best.pt"))
    if not models:
        return "Aucun modele 'best.pt' trouve."

    return "\n".join(str(path) for path in models)


def _safe_float(value: str | float | int | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_run_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and (
            (path / "results.csv").exists()
            or (path / "weights" / "best.pt").exists()
            or (path / "results.png").exists()
        ):
            candidates.append(path)
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def _read_results_csv(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_latest_run_figure(run_dir: Path):
    rows = _read_results_csv(run_dir)
    if not rows:
        return None

    epochs = [_safe_float(row.get("epoch")) for row in rows]
    box_loss = [_safe_float(row.get("train/box_loss")) for row in rows]
    cls_loss = [_safe_float(row.get("train/cls_loss")) for row in rows]
    dfl_loss = [_safe_float(row.get("train/dfl_loss")) for row in rows]
    map50 = [_safe_float(row.get("metrics/mAP50(B)")) for row in rows]
    recall = [_safe_float(row.get("metrics/recall(B)")) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, box_loss, label="box_loss")
    axes[0].plot(epochs, cls_loss, label="cls_loss")
    axes[0].plot(epochs, dfl_loss, label="dfl_loss")
    axes[0].set_title("Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, map50, label="mAP50")
    axes[1].plot(epochs, recall, label="Recall")
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(run_dir.name)
    fig.tight_layout()
    return fig


def _build_runs_overview_figure(run_dirs: list[Path]):
    points: list[tuple[str, float]] = []
    for run_dir in run_dirs[:8]:
        rows = _read_results_csv(run_dir)
        if not rows:
            continue
        last = rows[-1]
        score = _safe_float(last.get("metrics/mAP50(B)"))
        if score is not None:
            points.append((run_dir.name, score))

    if not points:
        return None

    labels = [item[0] for item in reversed(points)]
    values = [item[1] for item in reversed(points)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, values)
    ax.set_title("mAP50 des derniers runs")
    ax.set_xlabel("mAP50")
    fig.tight_layout()
    return fig


def inspect_runs(runs_root: str):
    root = Path(runs_root).expanduser()
    if not root.exists():
        return (
            f"Dossier introuvable: {root}",
            "Aucun modele.",
            None,
            None,
            None,
            None,
        )

    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        return (
            "Aucun run YOLO trouve.",
            list_models(runs_root),
            None,
            None,
            None,
            None,
        )

    latest = run_dirs[0]
    rows = _read_results_csv(latest)
    summary_lines = [f"Dernier run: {latest}"]
    if rows:
        last = rows[-1]
        summary_lines.extend(
            [
                f"Epoch finale: {last.get('epoch', '?')}",
                f"mAP50: {last.get('metrics/mAP50(B)', '?')}",
                f"Recall: {last.get('metrics/recall(B)', '?')}",
                f"Precision: {last.get('metrics/precision(B)', '?')}",
            ]
        )
    else:
        summary_lines.append("Pas de results.csv pour ce run.")

    results_png = latest / "results.png"
    confusion_png = latest / "confusion_matrix.png"
    latest_fig = _build_latest_run_figure(latest)
    overview_fig = _build_runs_overview_figure(run_dirs)
    return (
        "\n".join(summary_lines),
        list_models(runs_root),
        str(results_png) if results_png.exists() else None,
        str(confusion_png) if confusion_png.exists() else None,
        latest_fig,
        overview_fig,
    )


def run_prediction_visual(model_path: str, image_path: str, conf: float):
    if not image_path:
        return "Aucune image fournie.", None, pd.DataFrame()

    prediction = predict_image(model_path, Path(image_path).expanduser(), conf)
    detections = pd.DataFrame(prediction["detections"])
    return prediction["logs"], prediction["annotated_image"], detections


def build_ui() -> gr.Blocks:
    default_dataset = os.environ.get(
        "METEOR_ML_STUDIO_DATASET_DIR",
        str(PROJECT_ROOT / "data" / "dataset"),
    )
    default_runs = os.environ.get(
        "METEOR_ML_STUDIO_RUNS_DIR",
        str(PROJECT_ROOT / "runs"),
    )
    demo_image = resolve_demo_image()
    default_demo_image = str(demo_image) if demo_image else ""

    with gr.Blocks(title="Meteor ML Studio") as demo:
        gr.Markdown(
            """
            # Meteor ML Studio
            Interface locale pour piloter YOLO: dataset, entrainement, prediction
            et suivi visuel des runs.
            """
        )

        with gr.Tab("Dataset"):
            gr.Markdown(
                "Le dataset peut rester sur le Raspberry Pi. Ici, tu peux surtout "
                "preparer un dossier YOLO local de travail si tu veux en copier une partie."
            )
            dataset_root = gr.Textbox(
                label="Dossier du dataset",
                value=default_dataset,
            )
            class_name = gr.Textbox(label="Nom de classe", value="meteor")
            inspect_button = gr.Button("Inspecter le dataset")
            yaml_button = gr.Button("Generer dataset.yaml")
            dataset_logs = gr.Textbox(
                label="Logs / Resultat",
                lines=24,
                max_lines=30,
            )

            inspect_button.click(
                fn=run_script_stream,
                inputs=[
                    gr.State("dataset_tools.py"),
                    gr.State("inspect"),
                    dataset_root,
                ],
                outputs=dataset_logs,
            )
            yaml_button.click(
                fn=run_script_stream,
                inputs=[
                    gr.State("dataset_tools.py"),
                    gr.State("write-yaml"),
                    dataset_root,
                    class_name,
                ],
                outputs=dataset_logs,
            )

        with gr.Tab("Train"):
            train_dataset_yaml = gr.Textbox(
                label="Chemin vers dataset.yaml",
                value=str(Path(default_dataset) / "dataset.yaml"),
            )
            train_model = gr.Textbox(label="Modele de depart", value="yolo11n.pt")
            train_epochs = gr.Number(label="Epochs", value=50, precision=0)
            train_imgsz = gr.Number(label="Image size", value=640, precision=0)
            train_batch = gr.Number(label="Batch size", value=8, precision=0)
            train_project = gr.Textbox(label="Dossier des runs", value=default_runs)
            train_name = gr.Textbox(label="Nom du run", value="meteor_train")
            train_button = gr.Button("Lancer l'entrainement")
            train_logs = gr.Textbox(label="Logs d'entrainement", lines=24, max_lines=30)

            train_button.click(
                fn=run_script_stream,
                inputs=[
                    gr.State("train.py"),
                    gr.State("--data"),
                    train_dataset_yaml,
                    gr.State("--model"),
                    train_model,
                    gr.State("--epochs"),
                    train_epochs,
                    gr.State("--imgsz"),
                    train_imgsz,
                    gr.State("--batch"),
                    train_batch,
                    gr.State("--project"),
                    train_project,
                    gr.State("--name"),
                    train_name,
                ],
                outputs=train_logs,
            )

        with gr.Tab("Runs"):
            runs_root = gr.Textbox(label="Dossier des runs", value=default_runs)
            refresh_runs = gr.Button("Actualiser le suivi")
            run_summary = gr.Textbox(label="Resume du dernier run", lines=6, max_lines=10)
            models_output = gr.Textbox(label="Modeles trouves", lines=8, max_lines=12)
            with gr.Row():
                results_image = gr.Image(label="results.png", type="filepath")
                confusion_image = gr.Image(label="confusion_matrix.png", type="filepath")
            with gr.Row():
                latest_run_plot = gr.Plot(label="Courbes du dernier run")
                runs_overview_plot = gr.Plot(label="Vue d'ensemble des runs")

            refresh_runs.click(
                fn=inspect_runs,
                inputs=runs_root,
                outputs=[
                    run_summary,
                    models_output,
                    results_image,
                    confusion_image,
                    latest_run_plot,
                    runs_overview_plot,
                ],
            )

        with gr.Tab("Predict"):
            predict_model = gr.Textbox(label="Modele", value="yolo11n.pt")
            predict_image_input = gr.Image(
                label="Image a analyser",
                type="filepath",
                value=default_demo_image if default_demo_image else None,
            )
            predict_conf = gr.Number(label="Seuil de confiance", value=0.25)
            predict_button = gr.Button("Lancer la prediction")
            predict_logs = gr.Textbox(label="Logs de prediction", lines=20, max_lines=26)
            predict_result_image = gr.Image(label="Image annotee", type="filepath")
            predict_table = gr.Dataframe(label="Detections", interactive=False)

            predict_button.click(
                fn=run_prediction_visual,
                inputs=[predict_model, predict_image_input, predict_conf],
                outputs=[predict_logs, predict_result_image, predict_table],
            )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue().launch()
