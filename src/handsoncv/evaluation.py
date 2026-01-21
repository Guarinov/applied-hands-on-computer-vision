import torch
import os
import numpy as np
from torchvision.utils import save_image
from .utils import sample_flowers, sample_mnist, sample_unconditional
from .metrics import calculate_clip_score, calculate_fid, extract_inception_features

class Evaluator:
    """
    Utility class for evaluating a UNet-DDPM diffusion model under different
    conditioning regimes (unconditional, class-conditional, text-conditional).

    The evaluator supports:
    - Bottleneck embedding extraction via forward hooks
    - Image generation (unconditional, class-conditional, or text-conditional)
    - CLIP score computation (text-conditional only)
    - Classifier-based confidence and accuracy analysis (class-conditional only)
    - FID computation against real images (text-conditional only)

    Metrics are computed only when meaningful:
    - CLIP scores are computed if text conditioning is used (non-unconditional).
    - Classifier confidence/accuracy is computed if a classifier is provided and
      generation is class-conditional (non-text).
    - FID is computed if real data is provided and generation is text-conditional.

    Args:
        model (nn.Module): Trained UNet model used for diffusion sampling.
        ddpm (DDPM): Diffusion process handler.
        device (torch.device): Device for computation.
        clip_model (nn.Module, optional): CLIP model for text encoding and scoring.
        clip_preprocess (callable, optional): CLIP image preprocessing function.
        classifier (nn.Module, optional): Pretrained classifier for class-conditional evaluation.
        results_dir (str, optional): Directory to store generated images and embeddings.
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
    
    def _normalize_for_mnist_classifier(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Normalize generated images for MNIST classifier inference.
        
        Expects images in [-1, 1], rescales to [0, 1], then applies
        per-image min-max normalization followed by MNIST statistics.
        """
        # Apply image normalization and prepare for classifier
        # Rescale imgs from [-1,1] -> [0,1] for classifier
        imgs = (imgs + 1) / 2
        # Rescale each image in the batch so its own min is 0 and max is 1
        eps = 1e-8
        b = imgs.shape[0]
        flat = imgs.view(b, -1)
        img_min = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        img_max = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        imgs = (imgs - img_min) / (img_max - img_min + eps)

        # ! Apply Classifier-Specific One-Channel Normalization (MNIST Stats); please adjust if using a different dataset !
        imgs = (imgs - 0.1307) / 0.3081
        return imgs
    
    def evaluate_classifier_with_idk(self, classifier, x_gen, targets, device, idk_class: int, is_unconditional: bool):
        """
        Run classifier inference with IDK logic.

        Returns:
            clf_preds (list[str])
            clf_confidences (list[float])
            clf_statuses (list[str])
            stats (dict)
        """
        classifier.eval()

        clf_preds, clf_confidences, clf_statuses = [], [], []

        idk_count = 0
        total_correct = 0
        mistakes_avoided = 0
        missed_positive = 0

        with torch.no_grad():
            imgs = self._normalize_for_mnist_classifier(x_gen.to(device))
            logits = classifier(imgs)
            probs = torch.softmax(logits, dim=1)

            confidences, preds = torch.max(probs, dim=1)
            # For status logic
            digit_guesses = logits[:, :10].argmax(dim=1)

            for i in range(len(targets)):
                p = preds[i].item()
                conf = confidences[i].item()
                target = targets[i]

                is_idk = (p == idk_class)
                if is_idk:
                    idk_count += 1

                clf_preds.append("IDK" if is_idk else str(p))
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

        stats = {
            "idk_count": idk_count,
            "total_correct": total_correct,
            "mistakes_avoided": mistakes_avoided,
            "missed_positive": missed_positive,
            "avg_confidence": sum(clf_confidences) / len(clf_confidences)
                if is_unconditional and clf_confidences else 0.0,
        }

        return clf_preds, clf_confidences, clf_statuses, stats
    
    def compute_fid(self, x_gen, real_dataloader, device):
        """
        Compute FID score between real images and generated samples.

        Returns:
            fid_score (float | None)
        """
        if real_dataloader is None:
            return None

        from torch.utils.data import DataLoader, TensorDataset

        # Extract the 2048-dim Inception features from the real images (Input is [-1, 1])
        real_feats = extract_inception_features(real_dataloader, device)
        # Extract Generated Features (Input is [-1, 1] from DDPM)
        gen_loader = DataLoader(TensorDataset(x_gen), batch_size=32)
        gen_feats = extract_inception_features(gen_loader, device)

        return calculate_fid(real_feats, gen_feats)

    def run_full_evaluation(self, cond_list=None, real_dataloader=None, w_tests: list | None = None, n_unconditional=1000, idk_class=10):
        """
        Run a full evaluation pipeline for the diffusion model.

        Depending on the conditioning mode, this method performs:
        - Unconditional generation with classifier confidence analysis
        - Class-conditional generation with classifier confidence and accuracy
        - Text-conditional generation with CLIP score computation
        - Optional FID computation against real images (text-conditional only)

        Conditioning modes:
        - Unconditional: cond_list is None
        - Text-conditional: cond_list is a list of strings
        - Class-conditional: cond_list is a list of integer labels

        Metrics behavior:
        - CLIP scores are computed only for text-conditional (non-unconditional) generation.
        - Classifier confidence is computed if a classifier is provided and generation is
        class-conditional or unconditional (non-text).
        - Classification accuracy is computed only for class-conditional generation
        when ground-truth labels are available.
        - FID is computed only if real_dataloader is provided and generation is text-conditional.

        Args:
            cond_list (list[str] | list[int] | None): Conditioning inputs (text prompts or labels).
                If None, unconditional sampling is performed.
            real_dataloader (DataLoader, optional): Dataloader of real images for FID computation.
            w_tests (list[float], optional): Guidance weights for classifier-free guidance.
            n_unconditional (int, optional): Number of samples to generate in unconditional mode.
            idk_class (int, optional): Index of the IDK class in the classifier.

        Returns:
            tuple:
                results (list[dict]): Per-sample evaluation results including image paths,
                    embeddings, classifier outputs, and/or CLIP scores.
                fid_score (float | None): FID score if computed, otherwise None.
                final_acc (float): Classification accuracy excluding IDK (0 if not applicable).
                final_conf (float): Average classifier confidence for unconditional generation
                    (0 if not applicable).
        """
        final_acc = 0.0
        final_conf = 0.0
        is_unconditional = cond_list is None
        is_text_mode = isinstance(cond_list, list) and len(cond_list) > 0 and isinstance(cond_list[0], str)
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
        
        # Classifier evaluation 
        clf_preds, clf_confidences, clf_statuses, clf_stats = [], [], [], {}

        if self.classifier is not None and not is_text_mode:
            clf_preds, clf_confidences, clf_statuses, clf_stats = \
                self.evaluate_classifier_with_idk(
                    self.classifier,
                    x_gen,
                    eval_list,
                    self.device,
                    idk_class=idk_class,
                    is_unconditional=is_unconditional,
                )
        
        results = []
        # # Classifier Inference (IDK Logic)
        # clf_preds, clf_statuses, clf_confidences = [], [], []
        # idk_count = 0
        # mistakes_avoided = 0
        # missed_positive = 0
        # total_correct = 0
        total_samples = len(eval_list)
        
        # if self.classifier is not None and not is_text_mode:
        #     self.classifier.eval()
        #     with torch.no_grad():
        #         # Apply image normalization and prepare for classifier
        #         # Rescale imgs from [-1,1] -> [0,1] for classifier
        #         imgs = (x_gen.to(self.device) + 1) / 2
        #         # Rescale each image in the batch so its own min is 0 and max is 1
        #         eps = 1e-8
        #         b_size = imgs.shape[0]
        #         img_flat = imgs.view(b_size, -1)
        #         img_min = img_flat.min(dim=1, keepdim=True)[0].view(b_size, 1, 1, 1)
        #         img_max = img_flat.max(dim=1, keepdim=True)[0].view(b_size, 1, 1, 1)
        #         imgs = (imgs - img_min) / (img_max - img_min + eps)
                
        #         # ! Apply Classifier-Specific One-Channel Normalization (MNIST Stats); please adjust if using a different dataset !
        #         imgs = (imgs - 0.1307) / 0.3081
                
        #         logits = self.classifier(imgs)
        #         probs = torch.softmax(logits, dim=1)
        #         confidences, preds = torch.max(probs, dim=1)
        #         preds = logits.argmax(dim=1)
        #         # For status logic
        #         digit_guesses = logits[:, :10].argmax(dim=1)
                
        #         for i in range(total_samples):
        #             p = preds[i].item()
        #             conf = confidences[i].item()
        #             target = eval_list[i]
        #             is_idk = (p == idk_class)
        #             if is_idk: idk_count += 1
                    
        #             pred_label = "IDK" if is_idk else str(p)
        #             clf_preds.append(pred_label)
        #             clf_confidences.append(conf)
                    
        #             if is_unconditional:
        #                 status = f"{conf*100:.1f}% Conf"
        #             else:
        #                 if p == target:
        #                     total_correct += 1
        #                     status = "Correct"
        #                 elif is_idk:
        #                     # If we predicted IDK, would the digit guess have been wrong?
        #                     if digit_guesses[i].item() != target:
        #                         mistakes_avoided += 1
        #                         status = "Correct (IDK Avoided Mistake)"
        #                     else:
        #                         status = "Incorrect (Mistaken as IDK)"
        #                 else:
        #                     # Predicted a wrong digit (not IDK)
        #                     missed_positive += 1
        #                     status = "Incorrect"
        #             clf_statuses.append(status)

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
        # Uncomment the following line and comment out the line after if you want FID for conditional MNIST as well
        # if real_dataloader is not None and not is_unconditional:
        if real_dataloader is not None and is_text_mode:
            fid_score = self.compute_fid(x_gen, real_dataloader, self.device)
            
            # from torch.utils.data import DataLoader, TensorDataset
            # # Extract the 2048-dim Inception features from the real images (Input is [-1, 1])
            # real_feats = extract_inception_features(real_dataloader, self.device)
            # # Extract Generated Features (Input is [-1, 1] from DDPM)
            # gen_loader = DataLoader(TensorDataset(x_gen), batch_size=32) #batch_size=len(text_list))
            # gen_feats = extract_inception_features(gen_loader, self.device)
            # fid_score = calculate_fid(real_feats, gen_feats)
        
        # Global Accuracy (excluding IDK as per your provided function)
        print(f"\n--- Evaluation Results ({'Unconditional' if is_unconditional else 'Conditional'}) ---")
        
        
        # --- Reporting ---
        if self.classifier and not is_text_mode:
            print(f"Total samples: {total_samples}")
            print(f"IDK predictions: {clf_stats['idk_count']} "
                f"({100 * clf_stats['idk_count'] / total_samples:.2f}%)")

            if not is_unconditional:
                denom = total_samples - clf_stats["idk_count"]
                final_acc = clf_stats["total_correct"] / denom if denom > 0 else 0.0
                print(f"Accuracy (excluding IDK): {final_acc:.4f}")
                print(f"Mistakes avoided by IDK: {clf_stats['mistakes_avoided']}")
            else:
                final_conf = clf_stats["avg_confidence"]
                final_acc = 0.0
                
        # if self.classifier and not is_text_mode:
        #     print(f"Total samples: {total_samples}")
        #     print(f"IDK predictions: {idk_count} ({100*idk_count/total_samples:.2f}%)")
            
        #     if not is_unconditional:
        #         # Accuracy only makes sense when targets exist
        #         accuracy = total_correct / (total_samples - idk_count) if (total_samples - idk_count) > 0 else 0.0
        #         print(f"Accuracy (excluding IDK): {accuracy:.4f} ({100*accuracy:.2f}%)")
        #         print(f"Mistakes avoided by IDK: {mistakes_avoided}")
        #         print(f"Misclassified digits (non-IDK): {missed_positive}")
        #     else:
        #         # For unconditional, show the distribution of what the model is making
        #         from collections import Counter
        #         # dist = Counter(clf_preds)
        #         # print(f"Class Distribution: {dict(sorted(dist.items()))}")
        #         counts = Counter([p for p in clf_preds]) #
        #         print(f"Class Distribution(Predicted Classes):")
        #         for digit in sorted(counts.keys()):
        #             print(f"  Digit {digit}: {counts[digit]} samples")
                
        #         final_conf = sum(clf_confidences) / len(clf_confidences)
                
        # Return accuracy only if conditional
        # final_acc = (total_correct / (total_samples - idk_count)) if (not is_unconditional and (total_samples - idk_count) > 0) else 0.0
        # final_conf = final_conf if is_unconditional else 0.0
        return results, fid_score, final_acc, final_conf 