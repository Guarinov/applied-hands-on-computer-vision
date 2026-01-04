import torch
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image

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