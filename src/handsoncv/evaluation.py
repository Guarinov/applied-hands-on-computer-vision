import torch
import os
import numpy as np
from torchvision.utils import save_image
from .utils import sample_flowers, sample_mnist, sample_unconditional
from .metrics import calculate_clip_score, calculate_fid, extract_inception_features

class Evaluator:
    """
    Utility class for evaluating a UNet-DDPM diffusion model using CLIP and FID metrics.

    Captures intermediate embeddings via forward hooks, generates images from text prompts,
    computes CLIP scores, and optionally calculates FID against real images.

    Args:
        model (nn.Module): Trained UNet-DDPM model.
        ddpm (DDPM): Diffusion process object.
        clip_model (nn.Module, optional): Pretrained CLIP model for text-conditioning and scoring.
        clip_preprocess (callable, optional): CLIP preprocessing function for images for text-conditioning guidance.
        classifier (callable, optional): classifier to assess image generation quality for label-conditioning guidance.
        device (torch.device): Device for computation (CPU or CUDA).
        results_dir (str, optional): Directory to save generated images and embeddings. Defaults to "results".
    """
    def __init__(self, model, ddpm, device, clip_model=None, clip_preprocess=None, classifier=None, results_dir="results"):
        """Initialize evaluator and register forward hook for embedding extraction."""
        self.model = model
        self.ddpm = ddpm
        self.clip_model = clip_model #clip model
        self.clip_preprocess = clip_preprocess #clip tokenizer
        self.device = device
        self.results_dir = results_dir
        self.classifier = classifier #pre-trained MNIST classifier
        
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

    def run_full_evaluation(self, cond_list=None, real_dataloader=None, w_tests: list | None = None, n_unconditional=1000, idk_class=10):
        """
        Generates images from text prompts/integer labels, extracts bottleneck embeddings, 
        computes CLIP scores for prompts conditioning/classification accuracy for class guidance,
        and optionally evaluates FID against real images.

        Args:
            cond_list (list[str] | list[int]): List of text prompts or label indices for generation.
            real_dataloader (DataLoader, optional): Dataloader of real images for FID computation.
            w_tests (list[float] | None, optional): List of guidance weights to test during sampling.

        Returns:
            tuple:
                - results (list[dict]): List containing dictionaries with keys:
                    'prompt', 'img_path', 'clip_score', and 'embedding' (numpy array).
                - fid_score (float | None): FID score if real_dataloader is provided, else None.
        """
        
        is_unconditional = cond_list is None
        is_text_mode = False if is_unconditional else isinstance(cond_list[0], str)   
        os.makedirs(self.results_dir, exist_ok=True)                   
        
        # Generate images and extract embeddings via sample_flowers if is_text_mode or sample_mnist
        # Pass self.embeddings_storage so the hook output is captured
        if is_unconditional:
            # For unconditional, we generate a fixed number of samples
            x_gen = sample_unconditional(
                self.model, self.ddpm, n_unconditional, self.model.img_ch, 
                self.model.img_size, self.device, embeddings_storage=self.embeddings_storage
            )
            # Use dummy range for the loop
            eval_list = [None] * n_unconditional
        elif is_text_mode:
            x_gen, _ = sample_flowers(
                self.model, self.ddpm, self.clip_model, cond_list, 
                self.device, results_dir=self.results_dir,
                embeddings_storage=self.embeddings_storage, w_tests=w_tests
            )
            eval_list = cond_list
        else:
            x_gen, _ = sample_mnist(
                self.model, self.ddpm, cond_list, 
                self.device, results_dir=self.results_dir,
                embeddings_storage=self.embeddings_storage, w_tests=w_tests
            )
            eval_list = cond_list

        # Retrieve the extracted bottleneck embeddings
        # Shape: (B, 512, 8, 8) for Flowers or Shape (B, 10, 8, 8) for Mnist -> Flattened for FiftyOne
        extracted_embeddings = self.embeddings_storage['down2']
        
        results = []
        # Classifier Inference (IDK Logic)
        clf_preds, clf_statuses = [], []
        clf_confidences, clf_statuses = [], []
        idk_count = 0
        mistakes_avoided = 0
        missed_positive = 0
        total_correct = 0
        total_samples = len(eval_list)
        
        if self.classifier is not None and not is_text_mode:
            self.classifier.eval()
            with torch.no_grad():
                # Apply image normalization and prepare for classifier
                # Rescale imgs from [-1,1] -> [0,1] for classifier
                imgs = (x_gen.to(self.device) + 1) / 2
                # Rescale each image in the batch so its own min is 0 and max is 1
                eps = 1e-8
                b_size = imgs.shape[0]
                img_flat = imgs.view(b_size, -1)
                img_min = img_flat.min(dim=1, keepdim=True)[0].view(b_size, 1, 1, 1)
                img_max = img_flat.max(dim=1, keepdim=True)[0].view(b_size, 1, 1, 1)
                imgs = (imgs - img_min) / (img_max - img_min + eps)
                
                # ! Apply Classifier-Specific One-Channel Normalization (MNIST Stats); please adjust if using a different dataset !
                imgs = (imgs - 0.1307) / 0.3081
                
                logits = self.classifier(imgs)
                probs = torch.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)
                preds = logits.argmax(dim=1)
                # For status logic
                digit_guesses = logits[:, :10].argmax(dim=1)
                
                for i in range(total_samples):
                    p = preds[i].item()
                    conf = confidences[i].item()
                    target = eval_list[i]
                    is_idk = (p == idk_class)
                    if is_idk: idk_count += 1
                    
                    pred_label = "IDK" if is_idk else str(p)
                    clf_preds.append(pred_label)
                    clf_confidences.append(conf)
                    
                    if is_unconditional:
                        status = f"{conf*100:.1f}% Conf"
                    else:
                        if p == target:
                            total_correct += 1
                            status = "Correct"
                        elif is_idk:
                            # If we predicted IDK, would the digit guess have been wrong?
                            if digit_guesses[i].item() != target:
                                mistakes_avoided += 1
                                status = "Correct (IDK Avoided Mistake)"
                            else:
                                status = "Incorrect (Mistaken as IDK)"
                        else:
                            # Predicted a wrong digit (not IDK)
                            missed_positive += 1
                            status = "Incorrect"
                    clf_statuses.append(status)

        # Calcolate CLIP scores and save embeddings
        for i, cond in enumerate(eval_list):
            img_path = os.path.join(self.results_dir, f"gen_{i:03d}.png")
            # sample_mnist does not save the images 
            if is_unconditional:
                save_image(x_gen[i:i+1], img_path, normalize=True, value_range=(-1, 1))

            emb_vec = extracted_embeddings[i].view(-1).numpy()
            # Save embedding as .npy
            npy_path = img_path.replace(".png", ".npy")
            np.save(npy_path, emb_vec)

            # Calculate CLIP score if in Text Mode
            score = None
            if is_text_mode and self.clip_model:
                import clip as clip_lib
                score = calculate_clip_score(
                    img_path, cond, self.clip_model, self.clip_preprocess, clip_lib.tokenize, self.device
                )
                
            results.append({
                "condition": cond if cond is not None else "Unconditional",
                "img_path": os.path.abspath(img_path),
                "classifier_label": clf_preds[i] if clf_preds else "N/A",
                "status": clf_statuses[i] if clf_statuses else (score if score else "N/A"),
                "embedding": emb_vec
            })

        # Calculate FID if real data features are provided and is not unconditional
        fid_score = None
        if real_dataloader is not None and not is_unconditional:
            from torch.utils.data import DataLoader, TensorDataset
            # Extract the 2048-dim Inception features from the real images (Input is [-1, 1])
            real_feats = extract_inception_features(real_dataloader, self.device)
            # Extract Generated Features (Input is [-1, 1] from DDPM)
            gen_loader = DataLoader(TensorDataset(x_gen), batch_size=32) #batch_size=len(text_list))
            gen_feats = extract_inception_features(gen_loader, self.device)
            fid_score = calculate_fid(real_feats, gen_feats)
        
        # Global Accuracy (excluding IDK as per your provided function)
        print(f"\n--- Evaluation Results ({'Unconditional' if is_unconditional else 'Conditional'}) ---")
            
        if self.classifier and not is_text_mode:
            print(f"Total samples: {total_samples}")
            print(f"IDK predictions: {idk_count} ({100*idk_count/total_samples:.2f}%)")
            
            if not is_unconditional:
                # Accuracy only makes sense when targets exist
                accuracy = total_correct / (total_samples - idk_count) if (total_samples - idk_count) > 0 else 0.0
                print(f"Accuracy (excluding IDK): {accuracy:.4f} ({100*accuracy:.2f}%)")
                print(f"Mistakes avoided by IDK: {mistakes_avoided}")
                print(f"Misclassified digits (non-IDK): {missed_positive}")
            else:
                # For unconditional, show the distribution of what the model is making
                from collections import Counter
                # dist = Counter(clf_preds)
                # print(f"Class Distribution: {dict(sorted(dist.items()))}")
                counts = Counter([p for p in clf_preds]) #
                print(f"Class Distribution(Predicted Classes):")
                for digit in sorted(counts.keys()):
                    print(f"  Digit {digit}: {counts[digit]} samples")
                
                final_conf = sum(clf_confidences) / len(clf_confidences)
                
        # Return accuracy only if conditional
        final_acc = (total_correct / (total_samples - idk_count)) if (not is_unconditional and (total_samples - idk_count) > 0) else 0.0
        final_conf = final_conf if is_unconditional else 0.0
        return results, fid_score, final_acc, final_conf