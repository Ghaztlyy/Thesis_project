from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor


# ============================================================
# Thesis configuration
# ============================================================
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
IMG_SIZE = 224
DEFAULT_CLASS_NAMES = [
    "Coccidiosis",
    "Healthy",
    "New Castle Disease",
    "Salmonella",
]
SEVERITY_ORDER = ["Mild", "Moderate", "Severe"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

st.set_page_config(
    page_title="ConvNeXtV2 Thesis Evaluator",
    page_icon="🐔",
    layout="wide",
)


# ============================================================
# Utilities
# ============================================================
def normalized_name(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def interpolation_from_name(name: str) -> InterpolationMode:
    name = (name or "bicubic").lower()
    return {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }.get(name, InterpolationMode.BICUBIC)


def safe_torch_load(checkpoint_bytes: bytes, device: torch.device):
    """Prefer safer weights-only loading, then fall back for the thesis full-checkpoint format."""
    buffer = io.BytesIO(checkpoint_bytes)
    try:
        return torch.load(buffer, map_location=device, weights_only=True)
    except Exception:
        buffer.seek(0)
        return torch.load(buffer, map_location=device, weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        # Raw state_dict: every value should be tensor-like.
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Unsupported checkpoint format. Expected a thesis checkpoint containing "
        "'model_state_dict', a dict containing 'state_dict', or a raw PyTorch state_dict."
    )


def strip_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def checkpoint_class_names(checkpoint) -> list[str] | None:
    if isinstance(checkpoint, dict):
        names = checkpoint.get("class_names")
        if isinstance(names, (list, tuple)) and names:
            return [str(x) for x in names]
        signature = checkpoint.get("experiment_signature")
        if isinstance(signature, dict):
            names = signature.get("class_names")
            if isinstance(names, (list, tuple)) and names:
                return [str(x) for x in names]
    return None


def resolve_num_classes(state_dict: dict, fallback: int = 4) -> int:
    likely_keys = [
        "head.fc.weight",
        "head.weight",
        "classifier.weight",
        "fc.weight",
    ]
    for key in likely_keys:
        value = state_dict.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[0])
    for key, value in state_dict.items():
        if str(key).endswith(".weight") and torch.is_tensor(value) and value.ndim == 2:
            if int(value.shape[0]) <= 1000:
                return int(value.shape[0])
    return fallback


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    class_names: list[str]
    transform: Compose
    checkpoint_meta: dict
    device: torch.device


@st.cache_resource(show_spinner=False)
def load_checkpoint_model(
    checkpoint_bytes: bytes,
    uploaded_name: str,
    fallback_classes: tuple[str, ...],
    device_name: str,
) -> LoadedModel:
    device = torch.device(device_name)
    checkpoint = safe_torch_load(checkpoint_bytes, device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))

    names = checkpoint_class_names(checkpoint)
    inferred_classes = resolve_num_classes(state_dict, fallback=len(fallback_classes))
    if names is None:
        names = list(fallback_classes)
    if len(names) != inferred_classes:
        raise ValueError(
            f"Checkpoint appears to have {inferred_classes} output classes, but "
            f"{len(names)} class names were supplied/found."
        )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=inferred_classes,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint does not exactly match the expected ConvNeXtV2-Tiny architecture.\n\n"
            f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}\n\n"
            f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )

    model = model.to(device)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    mean = tuple(data_config["mean"])
    std = tuple(data_config["std"])
    interpolation = data_config.get("interpolation", "bicubic")

    transform = Compose(
        [
            Resize(
                (IMG_SIZE, IMG_SIZE),
                interpolation=interpolation_from_name(interpolation),
            ),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    meta = {
        "file": uploaded_name,
        "sha256": sha256_bytes(checkpoint_bytes),
        "model_name": checkpoint.get("model_name", MODEL_NAME)
        if isinstance(checkpoint, dict)
        else MODEL_NAME,
        "class_names": names,
        "input_size": IMG_SIZE,
        "mean": mean,
        "std": std,
        "interpolation": interpolation,
    }
    return LoadedModel(uploaded_name, model, names, transform, meta, device)


def pil_rgb(file_or_bytes) -> Image.Image:
    if hasattr(file_or_bytes, "read"):
        file_or_bytes.seek(0)
        image = Image.open(file_or_bytes)
    else:
        image = Image.open(io.BytesIO(file_or_bytes))
    return image.convert("RGB")


def predict_one(bundle: LoadedModel, image: Image.Image) -> dict:
    x = bundle.transform(image).unsqueeze(0).to(bundle.device)
    with torch.inference_mode():
        logits = bundle.model(x)
        probs = F.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())
    return {
        "index": idx,
        "label": bundle.class_names[idx],
        "confidence": float(probs[idx].item()),
        "probabilities": probs.detach().cpu().numpy(),
    }


def batch_predict(bundle: LoadedModel, images: list[Image.Image], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[int] = []
    confidences: list[float] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        x = torch.stack([bundle.transform(img) for img in chunk]).to(bundle.device)
        with torch.inference_mode():
            logits = bundle.model(x)
            probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        predictions.extend(pred.detach().cpu().tolist())
        confidences.extend(conf.detach().cpu().tolist())
    return np.asarray(predictions), np.asarray(confidences)


def gradcam(bundle: LoadedModel, image: Image.Image, class_index: int | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM from the output of the final ConvNeXtV2 stage."""
    model = bundle.model
    target_layer = model.stages[-1]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        x = bundle.transform(image).unsqueeze(0).to(bundle.device)
        logits = model(x)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        score = logits[0, class_index]
        score.backward()

        activation = captured["activation"]
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError("Gradients were not captured for Grad-CAM.")

        act = activation.detach()[0]
        grad = gradient.detach()[0]

        # ConvNeXt outputs are normally CxHxW. Handle HxWxC defensively.
        if act.ndim != 3:
            raise RuntimeError(f"Unexpected final-stage activation shape: {tuple(act.shape)}")
        channel_axis = int(np.argmax(act.shape))
        if channel_axis != 0:
            act = torch.movedim(act, channel_axis, 0)
            grad = torch.movedim(grad, channel_axis, 0)

        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0, keepdim=True).unsqueeze(0))
        cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        denom = cam.max().clamp_min(1e-8)
        cam = (cam / denom).detach().cpu().numpy()
        return cam, class_index
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(np.clip(heatmap, 0, 1))[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat_rgb, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def probability_table(class_names: list[str], probs: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"Class": class_names, "Probability": probs})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )


def metric_dict(y_true: Iterable[int], y_pred: Iterable[int], labels: list[int]) -> dict:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": float(accuracy),
        "Macro Precision": float(macro_precision),
        "Macro Recall": float(macro_recall),
        "Macro F1": float(macro_f1),
    }


def evaluate_thesis_metrics(
    frame: pd.DataFrame,
    pred_col: str,
    class_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    labels = list(range(len(class_names)))
    true_idx = frame["true_index"].to_numpy()
    pred_idx = frame[pred_col].to_numpy()
    overall = metric_dict(true_idx, pred_idx, labels)

    rows = []
    for severity in SEVERITY_ORDER:
        part = frame[frame["severity"] == severity]
        if part.empty:
            continue
        metrics = metric_dict(part["true_index"], part[pred_col], labels)
        rows.append({"Severity": severity, **metrics, "Support": len(part)})

    severity_df = pd.DataFrame(rows)
    severity_lookup = severity_df.set_index("Severity") if not severity_df.empty else pd.DataFrame()

    if all(sev in severity_lookup.index for sev in SEVERITY_ORDER):
        errors = [1.0 - float(severity_lookup.loc[sev, "Accuracy"]) for sev in SEVERITY_ORDER]
        overall["Mean Classification Error"] = float(np.mean(errors))
    else:
        overall["Mean Classification Error"] = math.nan

    if all(sev in severity_lookup.index for sev in ["Mild", "Severe"]):
        overall["Accuracy Robustness Drop"] = float(
            severity_lookup.loc["Mild", "Accuracy"] - severity_lookup.loc["Severe", "Accuracy"]
        )
        overall["Macro F1 Robustness Drop"] = float(
            severity_lookup.loc["Mild", "Macro F1"] - severity_lookup.loc["Severe", "Macro F1"]
        )
    else:
        overall["Accuracy Robustness Drop"] = math.nan
        overall["Macro F1 Robustness Drop"] = math.nan

    return overall, severity_df


def confusion_figure(y_true, y_pred, class_names: list[str], title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def parse_batch_zip(zip_bytes: bytes, class_names: list[str]) -> tuple[list[dict], list[str]]:
    class_map = {normalized_name(name): (idx, name) for idx, name in enumerate(class_names)}
    severity_map = {normalized_name(sev): sev for sev in SEVERITY_ORDER}
    # Accept common aliases used in the training notebook/folders.
    severity_map.update({"severity1": "Mild", "s1": "Mild"})
    severity_map.update({"severity2": "Moderate", "s2": "Moderate"})
    severity_map.update({"severity3": "Severe", "sever": "Severe", "s3": "Severe"})

    records: list[dict] = []
    skipped: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parts = [normalized_name(p) for p in path.parts[:-1]]
            class_hit = next((class_map[p] for p in parts if p in class_map), None)
            severity_hit = next((severity_map[p] for p in parts if p in severity_map), None)
            if class_hit is None or severity_hit is None:
                skipped.append(info.filename)
                continue
            try:
                data = zf.read(info)
                image = pil_rgb(data)
            except Exception:
                skipped.append(info.filename)
                continue
            class_idx, class_name = class_hit
            records.append(
                {
                    "filename": info.filename,
                    "severity": severity_hit,
                    "true_index": class_idx,
                    "true_label": class_name,
                    "image": image,
                }
            )
    return records, skipped


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


# ============================================================
# Header
# ============================================================
st.title("ConvNeXtV2 Thesis Model Evaluator")
st.caption(
    "Compare your standard/vanilla ConvNeXtV2 checkpoint against your sequential-curriculum checkpoint. "
    "Single-image mode provides predictions and Grad-CAM heatmaps; batch mode computes the thesis metrics."
)

with st.expander("What this app calculates", expanded=False):
    st.markdown(
        """
        **Single image**
        - Predicted disease class and confidence for both models
        - Per-class probabilities
        - Grad-CAM heatmap for each model

        **Labeled batch ZIP**
        - Accuracy
        - Macro Precision
        - Macro Recall
        - Macro F1-score
        - Mild / Moderate / Severe Accuracy and Macro F1
        - Accuracy Robustness Drop: `Accuracy_Mild - Accuracy_Severe`
        - Macro-F1 Robustness Drop: `MacroF1_Mild - MacroF1_Severe`
        - Mean Classification Error (study definition): mean of `1 - Accuracy_s` over Mild, Moderate, Severe
        - Confusion-matrix heatmaps
        """
    )


# ============================================================
# Model checkpoint setup
# ============================================================
# Expected folder structure:
# models/
# ├── run1/
# │   ├── vanilla.pth
# │   └── curriculum.pth
# ├── run2/
# │   ├── vanilla.pth
# │   └── curriculum.pth
# └── run3/
#     ├── vanilla.pth
#     └── curriculum.pth

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Run 1": {
        "vanilla": BASE_DIR / "models" / "run1" / "vanilla.pth",
        "curriculum": BASE_DIR / "models" / "run1" / "curriculum.pth",
    },
    "Run 2": {
        "vanilla": BASE_DIR / "models" / "run2" / "vanilla.pth",
        "curriculum": BASE_DIR / "models" / "run2" / "curriculum.pth",
    },
    "Run 3": {
        "vanilla": BASE_DIR / "models" / "run3" / "vanilla.pth",
        "curriculum": BASE_DIR / "models" / "run3" / "curriculum.pth",
    },
}

fallback_classes = tuple(DEFAULT_CLASS_NAMES)
device_name = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.header("Model Selection")
    selected_run = st.radio(
        "Select trained model run",
        list(MODEL_PATHS.keys()),
    )
    st.caption(
        "The corresponding Vanilla and Curriculum checkpoints are loaded automatically."
    )
    st.caption(f"Inference device: **{device_name.upper()}**")

vanilla_path = MODEL_PATHS[selected_run]["vanilla"]
curriculum_path = MODEL_PATHS[selected_run]["curriculum"]

missing_files = [
    str(path.relative_to(BASE_DIR))
    for path in (vanilla_path, curriculum_path)
    if not path.is_file()
]

if missing_files:
    st.error(
        f"Missing checkpoint file(s) for {selected_run}: "
        + ", ".join(missing_files)
    )
    st.info(
        "Add the weights under models/run1, models/run2, and models/run3 using "
        "the filenames vanilla.pth and curriculum.pth."
    )
    st.stop()

try:
    with st.spinner(f"Loading {selected_run} ConvNeXtV2 models..."):
        vanilla_bundle = load_checkpoint_model(
            vanilla_path.read_bytes(),
            vanilla_path.name,
            fallback_classes,
            device_name,
        )
        curriculum_bundle = load_checkpoint_model(
            curriculum_path.read_bytes(),
            curriculum_path.name,
            fallback_classes,
            device_name,
        )
except Exception as exc:
    st.error(f"Failed to load {selected_run} checkpoints.")
    st.exception(exc)
    st.stop()

if vanilla_bundle.class_names != curriculum_bundle.class_names:
    st.error(
        "The two checkpoints use different class orders. They cannot be compared safely.\n\n"
        f"Standard: {vanilla_bundle.class_names}\n\n"
        f"Curriculum: {curriculum_bundle.class_names}"
    )
    st.stop()

CLASS_NAMES = vanilla_bundle.class_names

st.success(f"{selected_run} checkpoints loaded successfully.")
meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.metric("Classes", len(CLASS_NAMES))
meta_col2.metric("Input size", f"{IMG_SIZE} × {IMG_SIZE}")
meta_col3.metric("Device", device_name.upper())
st.caption(f"Active model pair: {selected_run}")
st.caption("Class order: " + " → ".join(CLASS_NAMES))

single_tab, batch_tab = st.tabs(["Single Image + Grad-CAM", "Batch Evaluation + Thesis Metrics"])


# ============================================================
# Single-image mode
# ============================================================
with single_tab:
    st.subheader(f"Single-image comparison — {selected_run}")
    single_file = st.file_uploader(
        "Upload one poultry fecal image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        key="single_image",
    )

    if single_file is not None:
        image = pil_rgb(single_file)
        st.image(image, caption=single_file.name, width=360)

        vanilla_result = predict_one(vanilla_bundle, image)
        curriculum_result = predict_one(curriculum_bundle, image)

        left, right = st.columns(2)
        with left:
            st.markdown("### Standard / Vanilla")
            st.metric("Prediction", vanilla_result["label"])
            st.metric("Confidence", f"{vanilla_result['confidence'] * 100:.2f}%")
            vanilla_probs = probability_table(CLASS_NAMES, vanilla_result["probabilities"])
            st.dataframe(
                vanilla_probs.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(vanilla_probs.set_index("Class")["Probability"])

        with right:
            st.markdown("### Curriculum")
            st.metric("Prediction", curriculum_result["label"])
            st.metric("Confidence", f"{curriculum_result['confidence'] * 100:.2f}%")
            curriculum_probs = probability_table(CLASS_NAMES, curriculum_result["probabilities"])
            st.dataframe(
                curriculum_probs.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(curriculum_probs.set_index("Class")["Probability"])

        if vanilla_result["label"] == curriculum_result["label"]:
            st.success(f"Both models agree: **{vanilla_result['label']}**")
        else:
            st.warning(
                f"The models disagree — Standard: **{vanilla_result['label']}**, "
                f"Curriculum: **{curriculum_result['label']}**"
            )

        st.divider()
        st.subheader("Grad-CAM heatmaps")
        target_mode = st.radio(
            "Heatmap target",
            ["Each model's predicted class", "Choose one class for both models"],
            horizontal=True,
        )
        selected_target = None
        if target_mode == "Choose one class for both models":
            selected_name = st.selectbox("Target class", CLASS_NAMES)
            selected_target = CLASS_NAMES.index(selected_name)

        if st.button("Generate Grad-CAM", type="primary"):
            with st.spinner("Generating Grad-CAM heatmaps..."):
                v_cam, v_target = gradcam(vanilla_bundle, image, selected_target)
                c_cam, c_target = gradcam(curriculum_bundle, image, selected_target)
                v_overlay = overlay_heatmap(image, v_cam)
                c_overlay = overlay_heatmap(image, c_cam)

            h1, h2 = st.columns(2)
            with h1:
                st.image(
                    v_overlay,
                    caption=f"Standard / Vanilla — target: {CLASS_NAMES[v_target]}",
                    use_container_width=True,
                )
            with h2:
                st.image(
                    c_overlay,
                    caption=f"Curriculum — target: {CLASS_NAMES[c_target]}",
                    use_container_width=True,
                )
            st.caption(
                "Grad-CAM highlights spatial regions that most influenced the selected class score. "
                "It is an interpretability aid, not a quantitative robustness metric."
            )


# ============================================================
# Batch mode
# ============================================================
with batch_tab:
    st.subheader(f"Labeled batch evaluation — {selected_run}")
    st.markdown(
        "Upload a ZIP whose folder path contains **both the severity and the true class**. "
        "Severity/class order in the path does not matter."
    )
    st.code(
        """batch.zip
├── Mild
│   ├── Coccidiosis
│   ├── Healthy
│   ├── New Castle Disease
│   └── Salmonella
├── Moderate
│   └── ... same four classes ...
└── Severe
    └── ... same four classes ...""",
        language="text",
    )
    st.caption(
        "Aliases `severity1/s1`, `severity2/s2`, and `severity3/s3` are also recognized. "
        "The class folder names must match the checkpoint class names."
    )

    zip_file = st.file_uploader("Upload labeled batch ZIP", type=["zip"], key="batch_zip")
    batch_size = st.slider("Inference batch size", min_value=1, max_value=64, value=16, step=1)

    if zip_file is not None:
        try:
            records, skipped = parse_batch_zip(zip_file.getvalue(), CLASS_NAMES)
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")
            st.stop()

        if not records:
            st.error(
                "No labeled images could be parsed. Check that each image path contains a recognized "
                "severity folder and one of the four checkpoint class names."
            )
        else:
            preview = pd.DataFrame(
                [
                    {
                        "Filename": r["filename"],
                        "Severity": r["severity"],
                        "True class": r["true_label"],
                    }
                    for r in records
                ]
            )
            st.success(f"Parsed {len(records):,} labeled images.")
            if skipped:
                st.warning(f"Skipped {len(skipped):,} image files whose class/severity could not be determined.")
                with st.expander("Show skipped files"):
                    st.write(skipped[:500])

            count_table = (
                preview.groupby(["Severity", "True class"], observed=False)
                .size()
                .rename("Images")
                .reset_index()
            )
            st.markdown("#### Batch composition")
            st.dataframe(count_table, use_container_width=True, hide_index=True)

            if st.button("Run batch evaluation", type="primary"):
                images = [r["image"] for r in records]
                progress = st.progress(0, text="Running Standard / Vanilla inference...")
                v_pred, v_conf = batch_predict(vanilla_bundle, images, batch_size)
                progress.progress(50, text="Running Curriculum inference...")
                c_pred, c_conf = batch_predict(curriculum_bundle, images, batch_size)
                progress.progress(100, text="Evaluation complete.")

                result_df = pd.DataFrame(
                    {
                        "filename": [r["filename"] for r in records],
                        "severity": [r["severity"] for r in records],
                        "true_index": [r["true_index"] for r in records],
                        "true_label": [r["true_label"] for r in records],
                        "vanilla_pred_index": v_pred,
                        "vanilla_prediction": [CLASS_NAMES[i] for i in v_pred],
                        "vanilla_confidence": v_conf,
                        "curriculum_pred_index": c_pred,
                        "curriculum_prediction": [CLASS_NAMES[i] for i in c_pred],
                        "curriculum_confidence": c_conf,
                    }
                )

                vanilla_metrics, vanilla_by_severity = evaluate_thesis_metrics(
                    result_df,
                    "vanilla_pred_index",
                    CLASS_NAMES,
                )
                curriculum_metrics, curriculum_by_severity = evaluate_thesis_metrics(
                    result_df,
                    "curriculum_pred_index",
                    CLASS_NAMES,
                )

                st.session_state["batch_result_df"] = result_df
                st.session_state["vanilla_metrics"] = vanilla_metrics
                st.session_state["curriculum_metrics"] = curriculum_metrics
                st.session_state["vanilla_by_severity"] = vanilla_by_severity
                st.session_state["curriculum_by_severity"] = curriculum_by_severity

    if "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]
        vanilla_metrics = st.session_state["vanilla_metrics"]
        curriculum_metrics = st.session_state["curriculum_metrics"]
        vanilla_by_severity = st.session_state["vanilla_by_severity"]
        curriculum_by_severity = st.session_state["curriculum_by_severity"]

        st.divider()
        st.subheader("Thesis metric comparison")
        metric_order = [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Accuracy Robustness Drop",
            "Macro F1 Robustness Drop",
            "Mean Classification Error",
        ]
        comparison_df = pd.DataFrame(
            {
                "Metric": metric_order,
                "Standard / Vanilla": [vanilla_metrics[m] for m in metric_order],
                "Curriculum": [curriculum_metrics[m] for m in metric_order],
            }
        )
        formatted = comparison_df.copy()
        for col in ["Standard / Vanilla", "Curriculum"]:
            formatted[col] = formatted[col].map(format_metric)
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        st.caption(
            "For this app, Mean Classification Error follows the thesis study definition: "
            "the mean of (1 − Accuracy) across Mild, Moderate, and Severe. "
            "Robustness Drop compares Mild against Severe."
        )

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("#### Standard / Vanilla by severity")
            if not vanilla_by_severity.empty:
                display_v = vanilla_by_severity.copy()
                for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
                    display_v[col] = display_v[col].map(lambda x: f"{x:.2%}")
                st.dataframe(display_v, use_container_width=True, hide_index=True)
        with s2:
            st.markdown("#### Curriculum by severity")
            if not curriculum_by_severity.empty:
                display_c = curriculum_by_severity.copy()
                for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
                    display_c[col] = display_c[col].map(lambda x: f"{x:.2%}")
                st.dataframe(display_c, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Confusion-matrix heatmaps")
        severity_view = st.selectbox("Confusion matrix subset", ["Overall", *SEVERITY_ORDER])
        cm_df = result_df if severity_view == "Overall" else result_df[result_df["severity"] == severity_view]
        if cm_df.empty:
            st.info(f"No {severity_view} samples are present in this batch.")
        else:
            cm1, cm2 = st.columns(2)
            with cm1:
                fig = confusion_figure(
                    cm_df["true_index"],
                    cm_df["vanilla_pred_index"],
                    CLASS_NAMES,
                    f"Standard / Vanilla — {severity_view}",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with cm2:
                fig = confusion_figure(
                    cm_df["true_index"],
                    cm_df["curriculum_pred_index"],
                    CLASS_NAMES,
                    f"Curriculum — {severity_view}",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.divider()
        st.subheader("Per-image results")
        visible_cols = [
            "filename",
            "severity",
            "true_label",
            "vanilla_prediction",
            "vanilla_confidence",
            "curriculum_prediction",
            "curriculum_confidence",
        ]
        display_results = result_df[visible_cols].copy()
        st.dataframe(
            display_results.style.format(
                {
                    "vanilla_confidence": "{:.2%}",
                    "curriculum_confidence": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        metrics_export = {
            "class_names": CLASS_NAMES,
            "standard_vanilla": vanilla_metrics,
            "curriculum": curriculum_metrics,
            "standard_vanilla_by_severity": vanilla_by_severity.to_dict(orient="records"),
            "curriculum_by_severity": curriculum_by_severity.to_dict(orient="records"),
        }
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download per-image predictions CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="convnextv2_batch_predictions.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "Download thesis metrics JSON",
                data=json.dumps(metrics_export, indent=2, allow_nan=True).encode("utf-8"),
                file_name="convnextv2_thesis_metrics.json",
                mime="application/json",
            )
