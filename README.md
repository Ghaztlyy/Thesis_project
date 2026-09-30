# ConvNeXtV2 Thesis Model Evaluator

Streamlit application for comparing the thesis **Standard/Vanilla ConvNeXtV2** checkpoint against the **Sequential Curriculum ConvNeXtV2** checkpoint.

## Features

### Single image
- Upload one poultry fecal image.
- Run both checkpoints side-by-side.
- Show predicted class, confidence, and all class probabilities.
- Generate Grad-CAM heatmaps for each model.

### Labeled batch
Upload a ZIP whose image paths contain both a degradation severity and true disease class. The app computes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Severity-specific Accuracy / Macro Precision / Macro Recall / Macro F1
- Accuracy Robustness Drop: `Accuracy_Mild - Accuracy_Severe`
- Macro-F1 Robustness Drop: `MacroF1_Mild - MacroF1_Severe`
- Mean Classification Error (thesis study definition): mean of `1 - Accuracy_s` over Mild, Moderate, Severe
- Confusion-matrix heatmaps
- Per-image predictions and confidence
- CSV and JSON exports

## Expected checkpoint format

The app directly supports the full checkpoint format used by the training notebook:

```python
{
    "model_state_dict": ...,
    "model_name": ...,
    "class_names": ...,
    ...
}
```

It also supports a raw `state_dict` or a dictionary containing `state_dict`.

The architecture is fixed to:

```text
convnextv2_tiny.fcmae_ft_in22k_in1k
```

with a 224 x 224 input and four output classes.

## Labeled batch ZIP format

Recommended structure:

```text
batch.zip
├── Mild
│   ├── Coccidiosis
│   ├── Healthy
│   ├── New Castle Disease
│   └── Salmonella
├── Moderate
│   ├── Coccidiosis
│   ├── Healthy
│   ├── New Castle Disease
│   └── Salmonella
└── Severe
    ├── Coccidiosis
    ├── Healthy
    ├── New Castle Disease
    └── Salmonella
```

Class and severity may appear in either order in the path. Severity aliases `severity1/s1`, `severity2/s2`, and `severity3/s3` are recognized.

If your checkpoint contains `class_names`, those names and their exact output-index order are used automatically. Otherwise the app uses the fallback class order entered in the sidebar.

## Run locally

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

## GPU

The app automatically uses CUDA when PyTorch detects a compatible NVIDIA GPU; otherwise it runs on CPU.

If you want CUDA inference, install the PyTorch build appropriate for your CUDA environment before installing the remaining dependencies.

## Upload size

`.streamlit/config.toml` raises Streamlit's upload limit to 1 GB for larger evaluation ZIPs.

## Security note

Only load PyTorch checkpoint files you created or trust. Full `.pth` checkpoints may contain pickled Python data.
