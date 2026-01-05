import torch
import os
import numpy as np
from .utils import sample_flowers
from .metrics import calculate_clip_score, calculate_fid, extract_inception_features

class Evaluator:
    def __init__(self, model, ddpm, clip_model, clip_preprocess, device, results_dir="results"):
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
        def hook(model, input, output):
            # Capture the output of the layer (bottleneck)
            self.embeddings_storage[name] = output.detach().cpu()
        return hook

    def run_full_evaluation(self, text_list, real_dataloader=None):
        # Generate images and extract embeddings via sample_flowers
        # Pass self.embeddings_storage so the hook output is captured
        x_gen, _ = sample_flowers(
            self.model, self.ddpm, self.clip_model, text_list, 
            self.device, results_dir=self.results_dir, 
            embeddings_storage=self.embeddings_storage,
            w_tests=[2.0]
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
            # Extract the 2048-dim Inception features from the real images (Input is [0, 1])
            real_feats = extract_inception_features(real_dataloader, self.device, input_range_is_m1_1=False)
            # Extract Generated Features (Input is [-1, 1] from DDPM)
            gen_loader = DataLoader(TensorDataset(x_gen), batch_size=len(text_list))
            gen_feats = extract_inception_features(gen_loader, self.device, input_range_is_m1_1=False)
            fid_score = calculate_fid(real_feats, gen_feats)
        return results, fid_score