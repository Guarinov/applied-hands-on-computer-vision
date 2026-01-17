import torch
import os
import numpy as np
from .utils import sample_flowers
from .metrics import calculate_clip_score, calculate_fid, extract_inception_features

class Evaluator:
    """
    Utility class for evaluating a UNet-DDPM diffusion model using CLIP and FID metrics.

    Captures intermediate embeddings via forward hooks, generates images from text prompts,
    computes CLIP scores, and optionally calculates FID against real images.

    Args:
        model (nn.Module): Trained UNet-DDPM model.
        ddpm (DDPM): Diffusion process object.
        clip_model (nn.Module): Pretrained CLIP model for conditioning and scoring.
        clip_preprocess (callable): CLIP preprocessing function for images.
        device (torch.device): Device for computation (CPU or CUDA).
        results_dir (str, optional): Directory to save generated images and embeddings. Defaults to "results".
    """
    def __init__(self, model, ddpm, clip_model, clip_preprocess, device, results_dir="results"):
        """Initialize evaluator and register forward hook for embedding extraction."""
        self.model = model
        self.ddpm = ddpm
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.results_dir = results_dir
        
        # Setup Hook for Embedding Extraction after 'down2' downsampling module in UNet 
        self.embeddings_storage = {}
        self.model.down2.register_forward_hook(self._get_embedding_hook('down2'))

    def _get_embedding_hook(self, name):
        """
        Creates a forward hook to capture intermediate layer outputs.

        Args:
            name (str): Identifier for the layer whose output will be stored.

        Returns:
            function: A forward hook function to register in a PyTorch module.
        """
        def hook(model, input, output):
            # Capture the output of the layer (bottleneck)
            self.embeddings_storage[name] = output.detach().cpu()
        return hook

    def run_full_evaluation(self, text_list, real_dataloader=None, w_tests: list | None = None):
        """
        Generates images from text prompts, extracts bottleneck embeddings, computes CLIP scores,
        and optionally evaluates FID against real images.

        Args:
            text_list (list[str]): List of text prompts for generation.
            real_dataloader (DataLoader, optional): Dataloader of real images for FID computation.
            w_tests (list[float] | None, optional): List of guidance weights to test during sampling.

        Returns:
            tuple:
                - results (list[dict]): List containing dictionaries with keys:
                    'prompt', 'img_path', 'clip_score', and 'embedding' (numpy array).
                - fid_score (float | None): FID score if real_dataloader is provided, else None.
        """
        # Generate images and extract embeddings via sample_flowers
        # Pass self.embeddings_storage so the hook output is captured
        x_gen, _ = sample_flowers(
            self.model, self.ddpm, self.clip_model, text_list, 
            self.device, results_dir=self.results_dir, 
            embeddings_storage=self.embeddings_storage,
            w_tests=w_tests
        )

        # Retrieve the extracted bottleneck embeddings
        # Shape: (B, 512, 8, 8) -> Flattened for FiftyOne
        extracted_embeddings = self.embeddings_storage['down2']
        
        results = []
        for i, prompt in enumerate(text_list):
            img_path = os.path.join(self.results_dir, f"gen_{i:03d}.png")
            emb_vec = extracted_embeddings[i].view(-1).numpy()
            
            # Save embedding as .npy
            npy_path = img_path.replace(".png", ".npy")
            np.save(npy_path, emb_vec)

            # Calculate CLIP score
            import clip as clip_lib
            score = calculate_clip_score(
                img_path, prompt, self.clip_model, self.clip_preprocess, clip_lib.tokenize, self.device
            )

            results.append({
                "prompt": prompt,
                "img_path": os.path.abspath(img_path),
                "clip_score": score,
                "embedding": emb_vec
            })

        # Calculate FID if real data features are provided
        fid_score = None
        if real_dataloader is not None:
            from torch.utils.data import DataLoader, TensorDataset
            # Extract the 2048-dim Inception features from the real images (Input is [-1, 1])
            real_feats = extract_inception_features(real_dataloader, self.device)
            # Extract Generated Features (Input is [-1, 1] from DDPM)
            gen_loader = DataLoader(TensorDataset(x_gen), batch_size=32) #batch_size=len(text_list))
            gen_feats = extract_inception_features(gen_loader, self.device)
            fid_score = calculate_fid(real_feats, gen_feats)
        return results, fid_score