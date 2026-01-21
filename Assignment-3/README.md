# Denoising Probabilistic Diffusion Models

This repository explores **diffusion-based generative models** and **their evaluation** through a series of focused notebooks. It covers _text-to-image conditional diffusion with classifier-free guidance_,_ quantitative and embedding-based evaluation of generated samples_, and an _uncertainty-aware assessment pipeline using self-aware classifiers with `IDK` labels_. Together, these notebooks provide a practical, end-to-end framework for training, evaluating, and analyzing diffusion models and their outputs.

## 🗺️ Repository Map

| Path/File | Purpose |
|------|---------|
| `checkpoints/` | Stores the best model checkpoints (`*.pt`): two from the DDPM models trained in `05_a_*` and evaluated in `05_b_*`, and one from the bonus DDPM. Checkpoints are selected either by lowest validation loss (`*_best_*`) or highest CLIP score during training (`*_clip_best_*`). |
| `notebooks/` | Notebooks are intended to be run sequentially, from training the initial DDPM model to the final uncertainty-aware evaluation of generated MNIST images. |
| `results/` | Contains generated `.png` images and corresponding UNet bottleneck embeddings (`.npy`). Subfolders include `eval_01` (cropped TF Flowers), `eval_02` (MNIST generated with CFG, for experimentation), and `eval_02_unconditional` (unconditional MNIST generation). |
| `scripts/` | Code for downloading data from Google Drive public link. |

## 🐍 Setup Guide: Micromamba Environment

All dependencies and environment setup instructions are shared across assignments. Please, refer to the top-level [README.md](../README.md) for guidance on installing the `handsoncv` environment and required packages.

## 📥 Data Download for Analysis Reproducibility 

### 1. Cropped Flowers Dataset

This dataset is a cropped, reduced version of the TensorFlow (TF) flowers dataset, with only 3 labels (daisy, roses, sunflowers). Before starting the analysis, we need to download the dataset. This can be done by running the script:
```bash
python scripts/download_data.py
```
Inside the script, define the variable `URL` to point to the Google Drive folder containing the data. The folder must be organized as follows:
- Subfolders indicate the label type (daisy, roses pr sunflowers).
- Each sample is co-registered with a unique ID, e.g.:
  - daisy: `5673551_01d1ea993e_n.jpg`
  - roses: `12240303_80d87f77a3_n.jpg`
The data should be provided as a compressed .zip file for efficient downloading. During execution, the script creates a temporary directory to unzip the files, and the final dataset will be available in the following structure:

```text
data/cropped_flowers/
├── daisy/
│   ├── <no-series>.jpg 
│   ├── <no-series>.jpg
│   └── ...
├── roses/ 
│   ├── <no-series>.jpg 
│   ├── <no-series>.jpg 
│   └── ...       
└── sunflowers/
    ├── <no-series>.jpg 
    ├── <no-series>.jpg 
    └── ...   
```

2. MNIST Dataset (for Accuracy-vs-Coverage Analysis)

The MNIST dataset is used in notebooks `05_bonus_*`. It can be downloaded automatically using `torchvision.datasets`. The dataset is stored locally in the following structure:

```text
MNIST/raw/
├── t10k-images-idx3-ubyte
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx3-ubyte
├── t10k-labels-idx3-ubyte.gz
├── train-images-idx3-ubyte
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte
└── train-labels-idx1-ubyte.gz
```

## 📊 W\&B Projects Links

The runs associated with the analyses in `notebooks/` are available at the following public link [W&B Projects](https://wandb.ai/handsoncv-research/projects), under the username `guarino-vanessa-emanuela` for proper W&B access.

The trainings and evaluations for the different tasks in `Assignment-3` are organized in a single project: [diffusion-model-assessment-v2](https://wandb.ai/handsoncv-research/diffusion-model-assessment-v2?nw=nwuserguarinovanessaemanuela). This project contains runs with logged metrics such as:

- Conditional and unconditional image generations (media and table previews)
- Training and validation losses
- Average CLIP and FID scores
- Learning rate schedules
- GPU memory usage
- Model parameters
- Full configuration logs

Key Runs:

1. **Training of UNet-DDPM CFG on Cropped Flowers:** 
Run: [ddpm_unet_training](https://wandb.ai/handsoncv-research/diffusion-model-assessment-v2/runs/sk6jjg9p?nw=nwuserguarinovanessaemanuela)  
Description: UNet-DDPM model trained for text-prompt-to-image generation of flowers.

2. **Evaluation of UNet-DDPM CFG on Cropped Flowers:** 
Run: [ddpm_unet_evaluation](https://wandb.ai/handsoncv-research/handsoncv-maxpoolvsstride?nw=nwuserguarinovanessaemanuela)  
Description: Evaluates flower image generation using encoded CLIP prompts. Metrics include CLIP and FID scores, uniqueness, and representativeness, summarized in comparative tables on the project overview page: [diffusion-model-assessment-v2](https://wandb.ai/handsoncv-research/diffusion-model-assessment-v2?nw=nwuserguarinovanessaemanuela).

3. **Training of Unconditional UNet-DDPM on MNIST Digits:** 
Run: [ddpm_unet_mnist_trainingt](https://wandb.ai/handsoncv-research/handsoncv-cilp-assessment?nw=nwuserguarinovanessaemanuela)  
Description: UNet-DDPM model trained unconditionally to generate MNIST digits.

## 🃏 HuggingFace Datasets Repos

Two datasets containing generated images and associated embeddings/metrics are available on Hugging Face:

1. **TF Flowers Diffusion Assessment**  
   500 generated flower images with:  
   - UNet embeddings as field vectors  
   - FiftyOne's uniqueness and representativeness metrics  
   - CLIP score per sample  

   Downloadable as the dataset repository: [vanessaguarino/TFflowers-diffusion-assessment](https://huggingface.co/datasets/vanessaguarino/TFflowers-diffusion-assessment)

2. **MNIST Confidence-Thresholded Evaluation**  
   500 generated MNIST digits with:  
   - Pseudo-labels from the pretrained 11-class classifier  
   - Newly generated predictions using the optimized IDK cascade threshold  
   - UNet embeddings as field vectors  
   - FiftyOne's uniqueness and representativeness metrics  
   - UMAP representation of the UNet embeddings  

   Downloadable as the dataset repository: [vanessaguarino/mnist-confidence-thresholded-evaluation](https://huggingface.co/datasets/vanessaguarino/mnist-confidence-thresholded-evaluation)

---

### Uploading a New Dataset to Hugging Face

To upload newly generated images (with a modified procedure or a larger number of evaluated images), follow these steps:

#### Step 1 — Add YAML Metadata

Paste the following `YAML` header at the very top of the `README.md` inside the local export folder (`./data/hf_flower_evaluation_report`) to categorize the data as the `test` split:

```yaml
---
dataset_info:
  configs:
  - config_name: default
    data_files:
    - split: test
      path: "test/samples.json"
---
# Flower Diffusion Assessment Results
This dataset contains the FiftyOne export of the diffusion evaluation.
- **Split**: Test / Evaluation
- **Format**: FiftyOneDataset
```

#### Step 2 — Authenticate Hugging Face CLI

```bash
huggingface-cli login --token YOUR_HF_TOKEN
```

#### Step 3 — Create and Upload Dataset
Replace `your-username` with your actual HF username

```bash
# Create a new dataset repository. Usage: huggingface-cli upload [REPO_ID] [LOCAL_FOLDER] [REMOTE_FOLDER]
huggingface-cli repo create your-username/TFflowers-diffusion-assessment --repo-type dataset

# Upload the local export folder to the dataset by pointing either to the root directory using `.` or directly to the `test/` subfolder by using `test` as the upload path.
huggingface-cli upload your-username/TFflowers-diffusion-assessment ./data/hf_flower_evaluation_report test --repo-type dataset

# If the local dataset is too large, we recommend trying again with upload-large-folder
huggingface-cli upload-large-folder your-username/TFflowers-diffusion-assessment ./data/hf_flower_evaluation_report --repo-type dataset
```

## 🧾 Summary of Results

Below is a unified summary of all experiments, including the Training of UNet-DDPM CFG on Cropped Flowers (Notebook `05_a_*`), its Evaluation (Notebook `05_b_*`), and the Training & Uncertainty-Aware Evaluation of the Unconditional UNet-DDPM on MNIST Digits (Notebook `05_bonus_*`).

| Notebook | Experiment / Architecture      | Val Loss | Avg CLIP (%) | FID | Avg Classifier Conf (%) | Parameters | Sec / Epoch | GPU Mem (MB) |
|--------------|------------------------------------|--------------|------------------|-------|-------|----------------|------------------|------------------|
| 05_a         | Training UNet-DDPM CFG             | 0.0888       | 28.70            |   -   |   -   |   34,122,243   | 18.99            | 6310.46          |
| 05_b         | Eval UNet-DDPM CFG\*               | 0.0888       | 27.30            | 87.50 |   -   |   34,122,243   | -                | -                |
| 05_bonus     | Training Unconditional UNet-DDPM   | 0.0171       | -                |   -\* |   -   |   2,115,841    | 15.75            | 1109.53          |
| 05_bonus     | Eval Unconditional UNet-DDPM       | 0.0171       | -                |   -   | 81.79 |   2,115,841    | -                | -                |

\* Optimal guidance weight set to $3.0$.
\* FID score calculation is not recommended for single-channel images.

In-depth analyses of the metrics are contained in the indicated notebooks.

## 🔁 Reproducing the Main Results

Once the data have been downloaded and the `handsoncv` environment is installed, we are ready to reproduce all experiments.  
Select the `handsoncv` kernel in each notebook and execute the cells sequentially.

All experiments rely on the reusable functions and models provided in `src/`, which are installed in editable mode via `pip install -e .`.

---

### Workflow Overview

The notebooks are organized to follow a logical experimental progression, summarized below:

| Notebook | Purpose | Notes |
|--------|--------|-------|
| `05_a_*` | Train UNet-DDPM CFG on cropped TF Flowers               | Only needed if retraining is required         |
| `05_b_*` | Evaluate pretrained UNet-DDPM CFG on cropped TF Flowers | Metrics include CLIP score per sample, FID score, and FiftyOne’s uniqueness & representativeness |
| `05_bonus_*` | Evaluate Unconditional UNet-DDPM                    | Metrics include average classifier confidence and `IDK`-cascade thresholding using a pretrained 11-class |

---

### Reproducibility
To control randomness in experiments—including:
- dataset sampling,
- dataloader shuffling,
- model initialization,

set the `SEED` variable consistently across notebooks. This seed ensures reproducibility by controlling randomness in `numpy`, `torch`, and CUDA operations (`torch.cuda` and `torch.backends.cudnn`).  

- **Reproducing results:** Leave the preselected `SEED` unchanged to retrain and evaluate exactly as reported.  
- **Evaluation only:** If We do not wish to retrain, we can load the optimized model weights from the `checkpoints/` folder.  

**Notes on pretrained classifiers:**  
- The checkpoint for the pretrained $11$-class multiclass classifier (trained on MNIST digits + top $20\%$ hard/mistaken digits converted in `IDK`) is available at:`Assignment-1/checkpoints/level_20/best_multiclass.pth`. We can also experiment with other models trained on the top 2% and 10% of hard/mistaken digits converted in `IDK`.

--- 

### Running the Experiments
The notebooks follow a logical experimental progression but can be executed independently. Each notebook, along with the corresponding modules in `src/`, includes detailed inline documentation explaining: architectural choices, experimental settings, evaluation protocols.

As noted earlier:  

- Notebook `05_b_*`: Can be run directly if the relevant checkpoints are available in the respective folder.  
- Notebook `05_bonus_*`: Can also be run independently, provided the checkpoints from Assignment-1 are available. In this case, skip the cell titled *“Training...”* in Part 2 of the notebook.
---

## ⚠️ Limitations
As analyzed in notebook `05_b_*`, the DDPM-UNet CFG model trained on cropped versions of the TF-Flowers dataset captures only a limited aspect of the image-generation process due to the low image resolution and the small number of samples available.  

Reported training times, GPU memory usage, and performance metrics are hardware-dependent and may vary across different system configurations.  

Although random seeds are consistently set for NumPy, PyTorch, and CUDA, full determinism cannot be guaranteed due to non-deterministic GPU operations.
