# CILP Assessment: Multimodal Learning

This project explores multimodal machine learning using a synthetically generated dataset from **NVIDIA Omniverse**. The dataset consists of simple geometric objects—**cubes** and **spheres**—captured through two sensing modalities: RGB images and 2D LiDAR projections. The primary task is **binary classification** of object shape, leveraging complementary information from both modalities.

The workflow across the notebooks includes:

1. **Dataset Exploration and Preparation:**  
   - Construction of a **FiftyOne dataset** from the RGB and LiDAR images in `data/assessment/`.  
   - Alignment of RGB and LiDAR modalities by converting LiDAR beams to Cartesian coordinates and grouping paired samples.  
   - Creation of reproducible training and validation splits (30% validation), stored as `.json` for consistent use in subsequent experiments.  
   - Basic dataset statistics and visual inspection using the FiftyOne App.

2. **Multimodal Fusion Experiments:**  
   - Evaluation of **late fusion** (combining modalities at a later stage) and **intermediate fusion** (combining intermediate feature representations).  
   - Comparison of fusion strategies and architectural variants to identify the most effective model design.

3. **Ablation Studies:**  
   - Investigation of **convolutional downsampling strategies** in the `Embedder` network (`Strided Convolution` vs `Max Pooling`).  
   - Analysis of performance impact on classification accuracy, model complexity, and training efficiency.

4. **Cross-Modal Fine-Tuning (CILP):**  
   - Contrastive pretraining between RGB and LiDAR embeddings using a CLIP-like approach (`EfficientCILPModel`).  
   - Cross-modal projection from RGB to LiDAR embedding space.  
   - Fine-tuning of the combined pipeline to maximize RGB input accuracy relative to a LiDAR-only baseline.

This repository provides the full pipeline, including **data preparation**, **multimodal model training**, **fusion strategy evaluation**, and **cross-modal projection and fine-tuning**, serving as a hands-on benchmark for multimodal learning research.


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

