import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm
from PIL import Image

def extract_inception_features(dataloader, device):
    """Extracts 2048-dim features from the InceptionV3 pooling layer.
    Handles both 1-channel (MNIST) and 3-channel (Flowers) inputs.
    
    Args:
        dataloader: loader for images
        device: cpu/cuda
    """
    # Use weights='IMAGENET1K_V1' for modern torchvision
    model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
    model.fc = torch.nn.Identity() # Remove the final classification layer
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for batch in dataloader:
            # Batch is [B, 3, H, W] or [B, 1, H, W], assume in range [-1, 1]
            x = batch[0].to(device)   # x ∈ [-1, 1]
            
            # If greyscale images ([B, 1, H, W])
            if x.shape[1] == 1:
                # Repeat the single channel 3 times to satisfy InceptionV3 [B, 1, H, W] -> [B, 3, H, W]
                x = x.repeat(1, 3, 1, 1)
            
            # Resize
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
            
            # Normalize from [-1, 1] → [0, 1] (The FID Standard)
            x = (x + 1.0) / 2.0
            
            # Normalize as per ImageNet
            # Note: Apply manual normalization because transforms.Compose usually works on PIL
            x = x.clone() # to avoid in-place modification errors in some torch versions 
            x[:, 0] = (x[:, 0] - 0.485) / 0.229
            x[:, 1] = (x[:, 1] - 0.456) / 0.224
            x[:, 2] = (x[:, 2] - 0.406) / 0.225
            
            feat = model(x)
            features.append(feat.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def calculate_clip_score(image, text_prompt, model, preprocess, tokenizer, device):
    """
    Compute the CLIP similarity score between an image and a text prompt. The score is calculated as 
    the cosine similarity between the image and text features extracted by a CLIP model.

    Args:
        image (PIL.Image.Image or str): Input image, either as PIL Image or file path.
        text_prompt (str): Text prompt to compare against the image.
        model (torch.nn.Module): Preloaded CLIP model with `encode_image` and `encode_text` methods.
        preprocess (callable): CLIP image preprocessing function.
        tokenizer (callable): CLIP tokenizer for text inputs.
        device (torch.device): Device on which to run the computation.

    Returns:
        CLIP similarity score (float): Between the image and the text prompt.
            Higher values indicate greater semantic alignment.
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Preprocess inputs
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = tokenizer([text_prompt]).to(device)

    # Compute features and similarity
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item() # Calculate dot product
    return score

def calculate_fid(real_embeddings, gen_embeddings):
    """
    Compute the Frechet Inception Distance (FID) between two sets of embeddings. This implementation 
    assumes that embeddings are extracted from anInceptionV3 model (typically the pool3 layer).

    Args:
        real_embeddings (np.ndarray): Embeddings of real images of shape (N, 2048)
            extracted from an InceptionV3 model.
        gen_embeddings (np.ndarray): Embeddings of generated images of shape (N, 2048)
            extracted from an InceptionV3 model.

    Returns:
        Frechet Inception Distance (FID) score (float): Lower values indicate greater similarity 
            between real and generated image distributions.
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2) # Calculate sum squared difference between means
    covmean = sqrtm(sigma1.dot(sigma2))  # Calculate sqrt of product of covariances
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Final FID calculation
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_idk_metrics(correct_scores, incorrect_scores, thresholds = np.linspace(.5, .9, 100)):
    accuracies = []
    coverages = []
    
    total_samples = len(correct_scores) + len(incorrect_scores)
    
    for t in thresholds:
        # Which samples do we keep?
        accepted_correct = sum(1 for c in correct_scores if c >= t)
        accepted_incorrect = sum(1 for c in incorrect_scores if c >= t)
        
        total_accepted = accepted_correct + accepted_incorrect
        
        if total_accepted > 0:
            acc = accepted_correct / total_accepted
            cov = total_accepted / total_samples
        else:
            acc = 1.0 # If we accept nothing, we aren't "wrong"
            cov = 0.0
            
        accuracies.append(acc)
        coverages.append(cov)
        
    return thresholds, accuracies, coverages

def find_optimal_threshold(thresholds, accuracies, target_accuracy=0.9985):
    for t, acc in zip(thresholds, accuracies):
        if acc >= target_accuracy:
            return t
    return thresholds[-1]