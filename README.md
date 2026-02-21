# Applied Hands-On Computer Vision - HPI WiSe 2025

This repository contains the assessment results and shared codebase for the Hands-On-CV Research Seminar at the Hasso Plattner Institute (WiSe 2025-26).

This repository contains the following assignments:

1. **MNIST Dataset Curation Lab - `Assignment-1`**: Uses FiftyOne to audit label noise and implements a robust "I Don't Know" (IDK) classification strategy using LeNet-5 and Dynamic Focal Loss.
2. **CILP Assessment: Multimodal Learning - `Assignment-2`**: End-to-end study of multimodal fusion and cross-modal adaptation using synthetic RGB and LiDAR data for object classification.
3. \<tbd\>

The code in this repository builds upon the NVIDIA Lab Notebooks and in-class scratchpads. AI tools were used exclusively for bug fixing and improving readability, including enhancing class/function documentation, visualization styling, and docstrings, and were not used for full code generation or modeling decisions.

## 📂 Repository Structure
The project follows the "src-layout" to ensure shared logic is easily accessible across different assignments while keeping the root directory clean.

```text
Applied-Hands-On-Computer-Vision/
├── .gitignore               # Git ignore file
├── pyproject.toml           # Project metadata and dependencies
├── LICENSE                  # Project license
├── README.md                # Setup and usage instructions
├── src/
│   └── handsoncv/           # Main package
│       ├── __init__.py      # Logic specific to Assignment 2      
│       ├── datasets.py        
│       ├── models.py
│       ├── training.py
│       ├── utils.py
│       └── visualization.py       
├── tests/                   # Unit tests for src modules
│   ├── test_datasets.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_utils.py
│   └── test_visualization.py
├── Assignment-01/           # Folder for the first assessment
│   ├── requirements.txt
│   └── image-dataset-curation-with-specialist-models-LeNet5.ipynb     
└── Assignment-02/           # Folder for the second assessment
    ├── data/                # Local folder not synchronized with GitHub; reproducible with scripts/download_data.py
    ├── checkpoints/         # Saved model weights
    ├── notebooks/
    │   ├── 01_dataset_exploration.ipynb  # Task 1
    │   └── ...                            # Task 2+
    ├── results/             # Figures and tables
    ├── scripts/             # Data download script
    │   └── download_data.py
    └── subset.json          # .json file containing train/val splits, seed and metadata
```

Please refer to the README.md of the individual `Assignment-*` folder for detailed summaries of results, reproducibility instructions, and code organization.

## 🐍 Setup Guide: Micromamba Environment

This repository uses a virtual environment called `handsoncv`, created with **micromamba**, which is designed to be shared across the various assignments in this course. This guide will walk you through installing **micromamba** and setting up the `handsoncv` environment. This environment contains all the dependencies required to:

- Use the functions in the `src` folder.
- Run the analyses and experiments in the `notebooks` folder.

### 1. Install Micromamba

Micromamba is a tiny, fast, and standalone package manager that doesn't require a base Python installation.

#### macOS / Linux
Download the micromamba binary for your system architecture:

```bash
# Example: macOS Apple Silicon (arm64)
curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba
```

Verify the binary architecture:

```bash
file bin/micromamba
```

Install it,
```bash
sudo mv bin/micromamba /usr/local/bin/micromamba
```

Initialize micromamba,
```bash
micromamba shell init -s zsh -r ~/micromamba
source ~/.zshrc
```

For Linux x86_64, we replace the macOS URL with the Linux one

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```
The shell initialization then works identically as on MacOs. Restart your terminal after installation.

#### Windows (PowerShell)
On Windows, 
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
(handsoncv) pip install -e.
```

#### A Note on Assignment-1
The MNIST Dataset Curation Lab (`Assignment-1`) was originally executed using a standalone virtual environment by installing the dependencies listed in `requirements.txt`,
```bash
(Assignment-1) pip install -r requirements.txt
```
The assignment can also be re-run using the shared `handsoncv` environment, which provides all the required dependencies and ensures consistency across assignments.

### 2. Verify the Installation
To ensure PyTorch and other key packages are installed correctly:
```bash
(handsoncv) python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import fiftyone as fo; print('FiftyOne installed')"
```

If you are using VS Code or Jupyter Notebooks, ensure you select the handsoncv kernel. Since ipykernel is included in the dependencies, you can register it manually if it doesn't show up:
```bash
(handsoncv) python -m ipykernel install --user --name handsoncv --display-name "Python 3.11 (handsoncv)"
```
### 3. Run Tests 
Once the environment is set up, you can run unit tests for the src/handsoncv modules using pytest:
```bash
(handsoncv) pytest tests/ -v
```
This will execute all test files in the `tests/` folder and provide verbose output. Each `test_*.py` file should contain tests for the corresponding module in `src/handsoncv`.



