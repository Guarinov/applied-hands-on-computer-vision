# CILP Assessment: Multimodal Learning

This repository implements an end-to-end multimodal learning pipeline covering dataset preparation, multimodal fusion modeling, architectural ablation studies, and cross-modal fine-tuning using a CLIP-inspired approach. Synthetically generated RGB images and 2D LiDAR projections are jointly leveraged to study how complementary modalities can be fused, compared, and adapted across domains in a controlled experimental setting for foreground recognition of cubes and spheres.

## 🗺️ Repository Map

| Path/File | Purpose |
|------|---------|
| `checkpoints/` | Contains the 9 best model checkpoints (`*.pt`), corresponding to the analyses in notebooks `02_*`, `03_*`, and `04_*`. Each checkpoint was selected based on lowest validation loss during training. |
| `notebooks/` | Notebooks should be used **sequentially**, beginning with data split creation and visualization, and ending with cross-modal projection for domain adaptation to another modality. |
| `results/` | Folder containing: `.png` screenshots of the **FiftyOne App** from notebook `01_*`; **W&B dashboards** corresponding to each notebook training; visualizations of tables used for analyzing **fusion strategies** and related **ablation study**. |
| `scripts/` | Code for downloading data from Google Drive public link. |
| `subset_splits.json` | `.json` file contraining the train/validation splits for the dataset, along with the random seed (`42`) and the subset percentage (`0.3`) used for sampling. |

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

## 📊 W\&B Projects Links

The runs associated with the analyses in `notebooks/` are available at the following public link [W&B Projects](https://wandb.ai/handsoncv-research/projects), , under the username `guarino-vanessa-emanuela` for proper W&B access.

The trainings corresponding to the different tasks in **Assignment 2** are organized into three projects, each containing runs with logged metrics such as first validation predictions, training and validation losses, accuracy, learning rate schedule, GPU memory usage, checkpoints, model parameters, and full configuration logs:

1. **Fusion Architectures:** [handsoncv-fusion](https://wandb.ai/handsoncv-research/handsoncv-fusion?nw=nwuserguarinovanessaemanuela) – for the four fusion architectures implemented and compared.  
2. **Downsampling Ablation:** [handsoncv-maxpoolvsstride](https://wandb.ai/handsoncv-research/handsoncv-maxpoolvsstride?nw=nwuserguarinovanessaemanuela) – for comparing Max Pooling vs Strided Convolution in the `Embedder` class.  
3. **Cross-Modal Fine-Tuning (CILP):** [handsoncv-cilp-assessment](https://wandb.ai/handsoncv-research/handsoncv-cilp-assessment?nw=nwuserguarinovanessaemanuela) – for training components of the fine-tuned cross-modal classifier.

## 🧾 Summary of Results

Below is a unified summary of all experiments, including the Fusion Exploration (Notebook `02_*`), Ablation Study (Notebook `03_*`), and the CILP / Cross-Modal Pipeline (Notebook `04_*`).

| Notebook | Experiment / Architecture      | Val Loss | Accuracy\* (%) | Parameters | Sec / Epoch | GPU Mem (MB) |
|--------------|------------------------------------|--------------|------------------|----------------|------------------|------------------|
| 02           | Late Fusion                        | 0.01719      | 99.58            | 1,994,793      | 2.87             | 218.86           |
| 02           | Int Fusion (Concat)                | 0.00384      | 99.92            | 4,517,805      | 2.81             | 276.53           |
| 02           | Int Fusion (Add)                   | 0.00184      | 99.92            | 2,879,405      | 2.82             | 287.75           |
| 02           | Int Fusion (Mul)                   | 0.00916      | 99.75            | 2,879,405      | 2.84             | 310.61           |
| 03           | MaxPool (Baseline)                 | 0.00197      | 99.92            | 2,879,405      | 2.81             | 234.52           |
| 03           | Strided Convolution                | 0.01956      | 99.66            | 4,545,505      | 3.24             | 259.29          |
| 04           | LiDAR Classifier                   | 0.00093      | 99.92            | –              | 2.83             | 125.49           |
| 04           | CILP (Contrastive Pretraining)     | 0.16239      | N/A*             | –              | 2.83             | 219.59           |
| 04           | Cross-Modal Projector              | 0.27649      | N/A*             | –              | 2.76             | 261.32           |
| 04           | RGB → LiDAR Fine-Tuning            | 0.08399      | 97.38            | –              | 2.70             | 234.05           |

\* Accuracy is not applicable for contrastive and projection-only training stages.

## 🔁 Reproducing the Main Results

Once the data have been downloaded and the `handsoncv` environment is installed, you are ready to reproduce all experiments.  
Select the `handsoncv` kernel in each notebook and execute the cells sequentially.

All experiments rely on the reusable functions and models provided in `src/`, which are installed in editable mode via `pip install -e .`.

---

### Workflow Overview

The notebooks are organized to follow a logical experimental progression, summarized below:

| Notebook | Purpose | Notes |
|--------|--------|-------|
| `01_*` | Dataset exploration and split creation | Required only to regenerate dataset splits |
| `02_*` | Multimodal fusion experiments | Late and intermediate fusion strategies |
| `03_*` | Ablation study | Downsampling: MaxPool vs Strided Convolution |
| `04_*` | Cross-modal fine-tuning (CILP) | RGB → LiDAR projection and adaptation |

---

### Dataset Subsets and Reproducibility
- Notebook `01_*` **must be run** if you want to recreate the dataset splits.  
- The size of the subset can be controlled via the `PERCENTAGE_SUBSET` variable.  
- The provided `subset.json` file stores **30% of the original dataset** and is sufficient to reproduce all reported results.

To introduce different randomness in:
- dataset sampling,
- dataloader shuffling,
- model initialization,

modify the `SEED` variable consistently across notebooks. This seed controls `numpy`, `torch`, and CUDA-related randomness (`torch.cuda`, `torch.backends.cudnn`).

--- 

### Running the Experiments
- If the provided `subset.json` is used, notebooks `02_*`, `03_*`, and `04_*` can be run directly.
- These notebooks follow a **theoretical progression** but can be executed **independently**.
  
Each notebook and the corresponding modules in `src/` include detailed inline documentation explaining architectural choices, experimental settings, and evaluation protocols.

#### Notebook `04_*` (CILP and Cross-Modal Projection)
- The initial configuration and dataloader setup must be executed.
- Pretrained checkpoints for: the CILP-trained `Embedder`, the `CrossModalProjector`, and the `LidarClassifier` can be loaded directly to **evaluate the final `RGB2LiDARClassifier` model** without retraining. The same applies to the `CrossModalProjector` evaluation subsection.

---

### ⚠️ Limitations
The dataset used in this project is synthetically generated using NVIDIA Omniverse and consists of simple geometric objects, which makes it suitable for controlled benchmarking but limits direct generalization to real-world RGB–LiDAR scenarios. For efficiency and reproducibility, experiments are conducted on a fixed subset (30\%) of the original dataset, and results may differ when training on the full data. Reported training times, GPU memory usage, and performance metrics are hardware-dependent and may vary across different system configurations. Although random seeds are set consistently across NumPy, PyTorch, and CUDA, full determinism cannot be guaranteed due to non-deterministic GPU operations.
