# CILP Assessment: Multimodal Learning

This project explores multimodal machine learning using a synthetic dataset from **NVIDIA Omniverse**, consisting of simple geometric objects - **cubes** and **spheres** - captured as RGB images and 2D LiDAR projections. The main task is **binary classification** of object shape using complementary information from both modalities. The workflow includes:

1. **Dataset Exploration and Preparation**  
   - Build a **FiftyOne dataset** from the RGB and LiDAR images in `data/assessment/`.  
   - Align modalities by converting LiDAR beams to Cartesian coordinates and grouping paired samples.  
   - Create reproducible splits: **20% validation**, with a `.json` storing **30% of the original data** for consistency across notebooks.  
   - Compute dataset statistics and visually inspect samples using FiftyOne.

2. **Multimodal Fusion Experiments**  
   - Evaluate **late fusion** (combine modalities at a later stage) and **intermediate fusion** (combine intermediate features).  
   - Compare fusion strategies and architectural variants to identify optimal designs.

3. **Ablation Studies**  
   - Analyze **convolutional downsampling** in the `Embedder` network (`Strided Convolution` vs `Max Pooling`).  
   - Measure effects on accuracy, model size, and training efficiency.

4. **Cross-Modal Fine-Tuning (CILP)**  
   - Contrastive pretraining of RGB and LiDAR embeddings using a CLIP-like approach (`EfficientCILPModel`).  
   - Project RGB embeddings into LiDAR space and fine-tune the pipeline to maximize RGB accuracy.

This repository provides the full pipeline - **data preparation, multimodal training, fusion evaluation, and cross-modal fine-tuning** - as a hands-on benchmark for multimodal learning research.

## Repository Map

| Path | Purpose |
|------|---------|
| `checkpoints/` | Contains the 9 best model checkpoints (`*.pt`), corresponding to the analyses in notebooks `02_*`, `03_*`, and `04_*`. Each checkpoint was selected based on lowest validation loss during training. |
| `notebooks/` | Notebooks should be used **sequentially**, beginning with data split creation and visualization, and ending with cross-modal projection for domain adaptation to another modality. |
| `results/` | Code for downloading data from Google Drive public links  |
| `reproducibility/` | Lightweight scripts to reproduce key plots from the paper (e.g., Figure 4 and Figure 5b). |
| `output/` | Generated figures (`.png`, `.html`) and result tables in CSV format. |
| `tests/` | Unit tests (using PyTest) to ensure the reliability of core functionalities. |
| `environment.yml / pyproject.toml` | Environment and dependency specifications using Conda & Poetry. |


## 🐍 Setup Guide: Micromamba Environment

This repository uses a virtual environment called `handsoncv`, created with **micromamba**, which is designed to be shared across the various assignments in this course. This guide will walk you through installing **micromamba** and setting up the `handsoncv` environment. This environment contains all the dependencies required to:

- Use the functions in the `src` folder.
- Run the analyses and experiments in the `notebooks` folder.

### 1. Install Micromamba

Micromamba is a tiny, fast, and standalone package manager that doesn't require a base Python installation.

#### macOS / Linux
Run the following in your terminal:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
Restart your terminal after installation.

#### Windows (PowerShell)

```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)
```

### 2. Create \& Activate the Environment 
Install the project in editable mode to get all dependencies listed in pyproject.toml (including PyTorch ≥ 2.0, FiftyOne, etc.):
```bash
micromamba create -n handsoncv python=3.11 -c conda-forge
micromamba activate handsoncv
```

Install the project in editable mode to get all dependencies listed in `pyproject.toml` (including PyTorch ≥ 2.0, FiftyOne, etc.):
```bash
pip install -e.
```

### 2. Verify the Installation
To ensure PyTorch and other key packages are installed correctly:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import fiftyone as fo; print('FiftyOne installed')"
```

If you are using VS Code or Jupyter Notebooks, ensure you select the handsoncv kernel. Since ipykernel is included in the dependencies, you can register it manually if it doesn't show up:
```bash
python -m ipykernel install --user --name handsoncv --display-name "Python 3.11 (handsoncv)"
```

## 📥 Data Download for Analysis Reproducibility 

Before starting the analysis, we need to download the dataset. This can be done by running the script:
```bash
python scripts/download_data.py
```

Inside the script, define the variable `URL` to point to the Google Drive folder containing the Nvidia Omniverse RGB and LiDAR data. The folder should be organized as follows:
- Subfolders indicate the label type (e.g., cubes or spheres).
- Each sample is co-registered with a unique ID:
  - RGB: `0001.png`
  - LiDAR: `0001.npy`
The data should be provided as a compressed .zip file for efficient downloading. During execution, the script creates a temporary directory to unzip the files, and the final dataset will be available in the following structure:

```text
data/assessment/
├── cubes/
│   ├── rgb/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── lidar/
│       ├── 0000.npy
│       ├── 0001.npy
│       └── ...
└── spheres/
    ├── rgb/
    │   └── ...
    └── lidar/
        └── ...
```

