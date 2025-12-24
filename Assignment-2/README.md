# CILP Assessment: Multimodal Learning

## Setup Guide: Micromamba Environment

This guide explains how to install `micromamba` and set up the `handsoncv` development environment.

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

### 2. Create and Activate the Environment 
Navigate to the project root directory (where pyproject.toml is located) and follow these steps to create the environment with Python 3.11 and activate it:
```bash
micromamba create -n handsoncv python=3.11 -c conda-forge
micromamba activate handsoncv
```

Since we use pyproject.toml, we install the project in "editable" mode. This installs all the libraries listed in the configuration file (including PyTorch > 2.0, FiftyOne, etc.):

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