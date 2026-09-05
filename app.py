from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
load_dotenv(PROJECT_ROOT / ".env")


class TrainingProcessManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stop_requested = False

    def start(self, command: list[str]) -> subprocess.Popen[str]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Un entrainement est deja en cours.")
            self._stop_requested = False
            self._process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=os.name == "posix",
            )
            return self._process

    def stop_requested(self, process: subprocess.Popen[str]) -> bool:
        with self._lock:
            return self._process is process and self._stop_requested

    def request_stop(self) -> str:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return "Aucun entrainement n'est en cours."
            if self._stop_requested:
                return "L'arret de l'entrainement est deja en cours."
            self._stop_requested = True

        self._send_signal(process, signal.SIGINT)
        threading.Thread(
            target=self._ensure_stopped,
            args=(process,),
            daemon=True,
        ).start()
        return "Arret demande. YOLO termine proprement le run en cours."

    def clear(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._process is process:
                self._process = None
                self._stop_requested = False

    @staticmethod
    def _send_signal(process: subprocess.Popen[str], requested_signal: int) -> None:
        if process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, requested_signal)
        else:
            process.send_signal(requested_signal)

    def _ensure_stopped(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            self._send_signal(process, signal.SIGTERM)

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._send_signal(process, signal.SIGKILL)


TRAINING_MANAGER = TrainingProcessManager()
MAX_TRAIN_LOG_LINES = 1_000

STUDIO_THEME = gr.themes.Base(
    primary_hue="orange",
    secondary_hue="orange",
    neutral_hue="zinc",
    spacing_size="md",
    radius_size="sm",
    font=("Avenir Next", "Trebuchet MS", "sans-serif"),
    font_mono=("SFMono-Regular", "Menlo", "monospace"),
)

STUDIO_CSS = """
:root {
    --studio-bg: #101214;
    --studio-surface: #181b1f;
    --studio-surface-raised: #1e2227;
    --studio-border: #30353b;
    --studio-text: #f1eee7;
    --studio-muted: #a3a6aa;
    --studio-accent: #c96a3d;
    --studio-accent-hover: #db7a4d;
    --studio-danger: #b74d3d;
}

html {
    scroll-behavior: auto !important;
}

body,
.gradio-container {
    background: var(--studio-bg) !important;
    color: var(--studio-text) !important;
}

.gradio-container {
    position: relative;
    max-width: 1480px !important;
    padding: 28px clamp(18px, 4vw, 64px) 72px !important;
    isolation: isolate;
}

.gradio-container::before {
    position: fixed;
    inset: 0;
    z-index: -1;
    content: "";
    pointer-events: none;
    background:
        radial-gradient(circle at 82% 4%, rgba(201, 106, 61, 0.11), transparent 27rem),
        linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px);
    background-size: auto, 48px 48px, 48px 48px;
    mask-image: linear-gradient(to bottom, black, transparent 75%);
}

#studio-header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 48px;
    align-items: end;
    margin-bottom: 28px;
    padding: 26px 0 30px;
    border-bottom: 1px solid var(--studio-border);
    animation: studio-enter 520ms cubic-bezier(.2, .8, .2, 1) both;
}

.studio-kicker {
    display: flex;
    gap: 9px;
    align-items: center;
    margin-bottom: 12px;
    color: var(--studio-accent);
    font-family: "SFMono-Regular", Menlo, monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.studio-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--studio-accent);
    box-shadow: 0 0 0 5px rgba(201, 106, 61, 0.12);
    animation: studio-pulse 2.8s ease-in-out infinite;
}

#studio-header h1 {
    max-width: 900px;
    margin: 0;
    color: var(--studio-text);
    font-size: clamp(2.6rem, 6vw, 5.5rem);
    font-weight: 650;
    letter-spacing: -0.065em;
    line-height: 0.92;
}

#studio-header h1 span {
    color: var(--studio-accent);
}

#studio-header p {
    max-width: 680px;
    margin: 22px 0 0;
    color: var(--studio-muted);
    font-size: clamp(0.98rem, 1.5vw, 1.12rem);
    line-height: 1.65;
}

.studio-meta {
    display: grid;
    min-width: 170px;
    border-top: 1px solid var(--studio-border);
}

.studio-meta span {
    padding: 10px 2px;
    border-bottom: 1px solid var(--studio-border);
    color: var(--studio-muted);
    font-family: "SFMono-Regular", Menlo, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div[role="tablist"] {
    gap: 4px !important;
    margin: 0 0 24px !important;
    padding: 5px !important;
    border: 1px solid var(--studio-border) !important;
    border-radius: 8px !important;
    background: rgba(24, 27, 31, 0.88) !important;
    backdrop-filter: blur(14px);
}

button[role="tab"] {
    position: relative;
    min-height: 44px;
    border: 0 !important;
    border-radius: 5px !important;
    color: var(--studio-muted) !important;
    font-weight: 650 !important;
    transition: color 180ms ease, background-color 180ms ease, transform 180ms ease !important;
}

button[role="tab"]:hover {
    color: var(--studio-text) !important;
    background: rgba(255, 255, 255, 0.035) !important;
}

button[role="tab"][aria-selected="true"] {
    color: var(--studio-text) !important;
    background: var(--studio-surface-raised) !important;
    box-shadow: inset 0 -2px 0 var(--studio-accent) !important;
}

.studio-intro {
    max-width: 780px;
    margin: 4px 0 20px !important;
    color: var(--studio-muted) !important;
    line-height: 1.65;
}

.studio-panel {
    margin-bottom: 16px !important;
    padding: clamp(16px, 2vw, 24px) !important;
    border: 1px solid var(--studio-border) !important;
    border-radius: 9px !important;
    background: linear-gradient(145deg, rgba(30, 34, 39, 0.98), rgba(24, 27, 31, 0.98)) !important;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18) !important;
    animation: studio-enter 420ms 80ms cubic-bezier(.2, .8, .2, 1) both;
}

.studio-section-title h2,
.studio-section-title h3 {
    margin-bottom: 4px !important;
    color: var(--studio-text) !important;
    font-size: clamp(1.25rem, 2vw, 1.7rem) !important;
    letter-spacing: -0.03em;
}

.studio-section-title p {
    margin-top: 0 !important;
    color: var(--studio-muted) !important;
}

.gradio-container label span {
    color: #d4d1ca !important;
    font-size: 0.79rem !important;
    font-weight: 650 !important;
    letter-spacing: 0.015em;
}

.gradio-container input,
.gradio-container textarea {
    border-color: #343a41 !important;
    background: #14171a !important;
    color: var(--studio-text) !important;
    transition: border-color 160ms ease, box-shadow 160ms ease !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: var(--studio-accent) !important;
    box-shadow: 0 0 0 3px rgba(201, 106, 61, 0.13) !important;
}

.gradio-container button:not([role="tab"]) {
    min-height: 44px;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    transition: transform 180ms cubic-bezier(.2, .8, .2, 1),
        border-color 180ms ease, background-color 180ms ease, box-shadow 180ms ease !important;
}

.gradio-container button:not([role="tab"]):hover:not(:disabled) {
    transform: translateY(-1px);
}

.gradio-container button:not([role="tab"]):active:not(:disabled) {
    transform: scale(0.985);
}

.action-toggle button {
    min-height: 52px !important;
    border-color: transparent !important;
    background: var(--studio-accent) !important;
    color: #fff8f2 !important;
    box-shadow: 0 10px 28px rgba(201, 106, 61, 0.18) !important;
}

.action-toggle button:hover:not(:disabled) {
    background: var(--studio-accent-hover) !important;
    box-shadow: 0 14px 34px rgba(201, 106, 61, 0.24) !important;
}

.danger-action button {
    background: var(--studio-danger) !important;
    box-shadow: 0 10px 28px rgba(183, 77, 61, 0.2) !important;
}

.training-status textarea {
    min-height: 48px !important;
    color: #e6c1ae !important;
    font-family: "SFMono-Regular", Menlo, monospace !important;
    font-size: 0.78rem !important;
}

.terminal-output textarea {
    min-height: 420px !important;
    padding: 18px !important;
    border-color: #2f353b !important;
    background: #111417 !important;
    color: #c9d0c8 !important;
    font-family: "SFMono-Regular", Menlo, monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    caret-color: var(--studio-accent);
}

.studio-panel img,
.studio-panel canvas {
    border-radius: 6px !important;
}

@keyframes studio-enter {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes studio-pulse {
    0%, 100% { opacity: 0.62; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1); }
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 12px 12px 48px !important;
    }

    #studio-header {
        grid-template-columns: 1fr;
        gap: 24px;
        padding-top: 18px;
    }

    .studio-meta {
        grid-template-columns: repeat(3, 1fr);
    }

    .studio-meta span {
        text-align: center;
    }

    div[role="tablist"] {
        overflow-x: auto;
        flex-wrap: nowrap !important;
    }

    button[role="tab"] {
        min-width: 92px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""


def resolve_annotations_manifest() -> Path:
    annotations_root = PROJECT_ROOT / "data" / "annotations"
    manifests = sorted(
        annotations_root.glob("*/manifest.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if manifests:
        return manifests[0]
    return annotations_root / "positives" / "manifest.csv"


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


def _whole_number(value: float | int) -> str:
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Nombre entier attendu: {value}")
    return str(int(number))


def run_training_stream(
    dataset_yaml: str,
    model_path: str,
    epochs: float | int,
    image_size: float | int,
    batch_size: float | int,
    runs_root: str,
    run_name: str,
):
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "train.py"),
        "--data",
        str(dataset_yaml),
        "--model",
        str(model_path),
        "--epochs",
        _whole_number(epochs),
        "--imgsz",
        _whole_number(image_size),
        "--batch",
        _whole_number(batch_size),
        "--project",
        str(runs_root),
        "--name",
        str(run_name),
    ]
    try:
        process = TRAINING_MANAGER.start(command)
    except RuntimeError as error:
        yield (
            f"Erreur: {error}\n",
            gr.update(visible=True),
            gr.update(visible=False),
            str(error),
        )
        return

    command_log = "$ " + " ".join(command) + "\n"
    yield (
        command_log,
        gr.update(visible=False),
        gr.update(visible=True, interactive=True, value="Arreter l'entrainement"),
        "Demarrage de l'entrainement...",
    )

    lines: list[str] = []
    logs_truncated = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if len(lines) > MAX_TRAIN_LOG_LINES:
                lines = lines[-MAX_TRAIN_LOG_LINES:]
                logs_truncated = True
            prefix = "[Les lignes les plus anciennes ont ete masquees.]\n" if logs_truncated else ""
            yield (
                prefix + "".join(lines),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

        return_code = process.wait()
        if TRAINING_MANAGER.stop_requested(process):
            lines.append("\n[Entrainement arrete proprement.]\n")
            final_status = "Entrainement arrete proprement."
        else:
            lines.append(f"\n[exit code: {return_code}]\n")
            final_status = (
                "Entrainement termine."
                if return_code == 0
                else f"L'entrainement s'est termine avec le code {return_code}."
            )
        yield (
            "".join(lines[-MAX_TRAIN_LOG_LINES:]),
            gr.update(visible=True),
            gr.update(visible=False),
            final_status,
        )
    except Exception as error:
        lines.append(f"\n[Erreur: {error}]\n")
        yield (
            "".join(lines[-MAX_TRAIN_LOG_LINES:]),
            gr.update(visible=True),
            gr.update(visible=False),
            f"Erreur pendant l'entrainement: {error}",
        )
    finally:
        if process.poll() is None:
            TRAINING_MANAGER.request_stop()
        else:
            TRAINING_MANAGER.clear(process)


def stop_training():
    message = TRAINING_MANAGER.request_stop()
    return message, gr.update(interactive=False, value="Arret en cours...")


def configure_log_autoscroll(enabled: bool):
    return gr.update(autoscroll=enabled)


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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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
    import pandas as pd

    from scripts.predict import predict_image

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
    default_annotations_manifest = os.environ.get(
        "METEOR_ML_STUDIO_ANNOTATIONS_MANIFEST",
        str(resolve_annotations_manifest()),
    )
    default_negatives = os.environ.get(
        "METEOR_ML_STUDIO_NEGATIVES_DIR",
        str(PROJECT_ROOT / "data" / "raw" / "false_positives"),
    )
    demo_image = resolve_demo_image()
    default_demo_image = str(demo_image) if demo_image else ""

    with gr.Blocks(title="Meteor ML Studio") as demo:
        gr.HTML(
            """
            <header id="studio-header">
                <div>
                    <div class="studio-kicker">
                        <span class="studio-status-dot"></span>
                        Atelier de vision local
                    </div>
                    <h1>Meteor <span>ML Studio</span></h1>
                    <p>
                        Prepare le dataset, entraine YOLO et controle chaque run
                        depuis un espace de travail unique.
                    </p>
                </div>
                <div class="studio-meta" aria-label="Configuration du studio">
                    <span>YOLO 11</span>
                    <span>1 classe</span>
                    <span>Execution locale</span>
                </div>
            </header>
            """
        )

        with gr.Tab("Dataset"):
            gr.Markdown(
                "Construis ici le dataset YOLO utilise pour l'entrainement. Les images "
                "positives viennent du manifeste d'annotations et les faux positifs "
                "sont selectionnes dans le dossier local.",
                elem_classes=["studio-intro"],
            )
            with gr.Group(elem_classes=["studio-panel"]):
                gr.Markdown(
                    "## Sources et repartition\nIndique ou trouver les images avant de construire le lot YOLO.",
                    elem_classes=["studio-section-title"],
                )
                with gr.Row():
                    dataset_root = gr.Textbox(
                        label="Dossier du dataset",
                        value=default_dataset,
                        scale=3,
                    )
                    class_name = gr.Textbox(
                        label="Nom de classe",
                        value="meteor",
                        scale=1,
                    )
                annotations_manifest = gr.Textbox(
                    label="Manifeste des annotations positives",
                    value=default_annotations_manifest,
                )
                negatives_dir = gr.Textbox(
                    label="Dossier des faux positifs",
                    value=default_negatives,
                )
                with gr.Row():
                    max_negatives = gr.Number(
                        label="Nombre maximal de faux positifs",
                        value=600,
                        precision=0,
                    )
                    val_percent = gr.Number(
                        label="Part reservee a la validation (%)",
                        value=20,
                        precision=0,
                    )
                    split_seed = gr.Number(
                        label="Graine de repartition",
                        value=42,
                        precision=0,
                    )
                build_button = gr.Button(
                    "Construire le dataset",
                    variant="primary",
                    elem_classes=["action-toggle"],
                )
                with gr.Row():
                    inspect_button = gr.Button("Inspecter le dataset")
                    yaml_button = gr.Button("Regenerer uniquement dataset.yaml")
            dataset_logs = gr.Textbox(
                label="Logs / Resultat",
                lines=24,
                max_lines=30,
                elem_classes=["terminal-output"],
            )

            build_button.click(
                fn=run_script_stream,
                inputs=[
                    gr.State("dataset_tools.py"),
                    gr.State("build"),
                    dataset_root,
                    annotations_manifest,
                    negatives_dir,
                    gr.State("--max-negatives"),
                    max_negatives,
                    gr.State("--val-percent"),
                    val_percent,
                    gr.State("--seed"),
                    split_seed,
                    gr.State("--class-name"),
                    class_name,
                ],
                outputs=dataset_logs,
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
            gr.Markdown(
                "Configure un run, suis sa progression sans perdre ta position dans "
                "la page et interromps-le proprement si necessaire.",
                elem_classes=["studio-intro"],
            )
            with gr.Group(elem_classes=["studio-panel"]):
                gr.Markdown(
                    "## Configuration du run\nLes valeurs ci-dessous sont transmises directement a YOLO.",
                    elem_classes=["studio-section-title"],
                )
                with gr.Row():
                    train_dataset_yaml = gr.Textbox(
                        label="Chemin vers dataset.yaml",
                        value=str(Path(default_dataset) / "dataset.yaml"),
                        scale=3,
                    )
                    train_model = gr.Textbox(
                        label="Modele de depart",
                        value="yolo11n.pt",
                        scale=1,
                    )
                with gr.Row():
                    train_epochs = gr.Number(label="Epochs", value=50, precision=0)
                    train_imgsz = gr.Number(label="Image size", value=640, precision=0)
                    train_batch = gr.Number(label="Batch size", value=8, precision=0)
                with gr.Row():
                    train_project = gr.Textbox(
                        label="Dossier des runs",
                        value=default_runs,
                        scale=3,
                    )
                    train_name = gr.Textbox(
                        label="Nom du run",
                        value="meteor_train",
                        scale=1,
                    )
                train_autoscroll = gr.Checkbox(
                    label="Faire defiler automatiquement les logs",
                    value=False,
                )
            train_button = gr.Button(
                "Lancer l'entrainement",
                variant="primary",
                elem_classes=["action-toggle"],
            )
            stop_train_button = gr.Button(
                "Arreter l'entrainement",
                variant="stop",
                visible=False,
                elem_classes=["action-toggle", "danger-action"],
            )
            train_status = gr.Textbox(
                label="Etat",
                value="Aucun entrainement lance depuis cette session.",
                interactive=False,
                lines=1,
                elem_classes=["training-status"],
            )
            train_logs = gr.Textbox(
                label="Logs d'entrainement",
                lines=24,
                max_lines=30,
                autoscroll=False,
                elem_id="training-logs",
                elem_classes=["terminal-output"],
            )

            train_button.click(
                fn=run_training_stream,
                inputs=[
                    train_dataset_yaml,
                    train_model,
                    train_epochs,
                    train_imgsz,
                    train_batch,
                    train_project,
                    train_name,
                ],
                outputs=[
                    train_logs,
                    train_button,
                    stop_train_button,
                    train_status,
                ],
                scroll_to_output=False,
                show_progress="hidden",
            )
            stop_train_button.click(
                fn=stop_training,
                outputs=[train_status, stop_train_button],
                queue=False,
                scroll_to_output=False,
                show_progress="hidden",
            )
            train_autoscroll.change(
                fn=configure_log_autoscroll,
                inputs=train_autoscroll,
                outputs=train_logs,
                queue=False,
                scroll_to_output=False,
                show_progress="hidden",
            )

        with gr.Tab("Runs"):
            gr.Markdown(
                "Compare les derniers runs et retrouve rapidement le modele a tester.",
                elem_classes=["studio-intro"],
            )
            with gr.Group(elem_classes=["studio-panel"]):
                runs_root = gr.Textbox(label="Dossier des runs", value=default_runs)
                refresh_runs = gr.Button("Actualiser le suivi", variant="primary")
                with gr.Row():
                    run_summary = gr.Textbox(
                        label="Resume du dernier run",
                        lines=6,
                        max_lines=10,
                    )
                    models_output = gr.Textbox(
                        label="Modeles trouves",
                        lines=8,
                        max_lines=12,
                    )
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
            gr.Markdown(
                "Charge une image et observe les detections produites par le modele choisi.",
                elem_classes=["studio-intro"],
            )
            with gr.Group(elem_classes=["studio-panel"]):
                with gr.Row():
                    predict_model = gr.Textbox(
                        label="Modele",
                        value="yolo11n.pt",
                        scale=3,
                    )
                    predict_conf = gr.Number(
                        label="Seuil de confiance",
                        value=0.25,
                        scale=1,
                    )
                predict_image_input = gr.Image(
                    label="Image a analyser",
                    type="filepath",
                    value=default_demo_image if default_demo_image else None,
                )
                predict_button = gr.Button("Lancer la prediction", variant="primary")
            predict_logs = gr.Textbox(
                label="Logs de prediction",
                lines=20,
                max_lines=26,
                elem_classes=["terminal-output"],
            )
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
    app.queue().launch(theme=STUDIO_THEME, css=STUDIO_CSS)
