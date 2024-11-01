import torch
import torchvision
import os
import argparse
from improved_precision_recall import IPR
from functools import partial
import torch.nn.functional as F

from model import Generator
from utils import load_model
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from variables import *
from tqdm import tqdm, trange
from pathlib import Path
from PIL import Image

import model_vgg.vgg_5 as model_vgg
vgg5 = model_vgg.VGG().to(device).eval()
vgg5.load_state_dict(torch.load("model_vgg/VGG_fine_tuned.pth"))
batch_size = 16

def vgg5_encoding(images):
    """
    Extract features of vgg16-fc2 for all images
    params:
        images: torch.Tensors of size N x C x H x W
    returns:
        A numpy array of dimension (num images, dims)
    """
    desc = 'extracting features of %d images' % images.size(0)
    num_batches = int(np.ceil(images.size(0) / batch_size))
    images = images.reshape(-1, 28, 28).unsqueeze(1)#.repeat(1, 3, 1, 1)
    #resize = partial(F.interpolate, size=(224, 224))

    features = []
    for bi in trange(num_batches, desc=desc):
        start = bi * batch_size
        end = start + batch_size
        batch = images[start:end].float()
        #batch = resize(batch).float()
        before_fc = vgg5.features(batch.to(device))
        #before_fc = before_fc.view(-1, 7 * 7 * 512)
        feature = vgg5.classifier[:4](before_fc)
        features.append(feature.cpu().data.numpy())

    return np.concatenate(features, axis=0)

import torchvision.models as models
vgg16 = models.vgg16(pretrained=True).to(device).eval()
batch_size = 16

def vgg16_encoding(images):
    """
    Extract features of vgg16-fc2 for all images
    params:
        images: torch.Tensors of size N x C x H x W
    returns:
        A numpy array of dimension (num images, dims)
    """
    desc = 'extracting features of %d images' % images.size(0)
    num_batches = int(np.ceil(images.size(0) / batch_size))
    images = images.reshape(-1, 28, 28).unsqueeze(1).repeat(1, 3, 1, 1)
    resize = partial(F.interpolate, size=(224, 224))

    features = []
    for bi in trange(num_batches, desc=desc):
        start = bi * batch_size
        end = start + batch_size
        batch = images[start:end]
        batch = resize(batch).float()
        before_fc = vgg16.features(batch.to(device))
        before_fc = before_fc.view(-1, 7 * 7 * 512)
        feature = vgg16.classifier[:4](before_fc)
        features.append(feature.cpu().data.numpy())

    return np.concatenate(features, axis=0)

def tsne_pipeline(embeddings,labels, title):
    """
    Runs TSNE on the provided embeddings and visualizes the result.
    
    Parameters:
    embeddings (np.ndarray): Array of data embeddings, shape (n_samples, n_features)
    title (str): Title for the plot
    """
    tsne = TSNE(n_components=2, random_state=42,verbose=1)
    X_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=2)
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()

def load_bw_images(directory_path):
    image_paths = list(Path(directory_path).glob('*.[pj][np][g]*'))  # matches .jpg, .jpeg, .png
    if not image_paths:
        raise ValueError(f"No images found in {directory_path}")
    
    images = []
    for img_path in image_paths:
        # Open image and convert to grayscale
        with Image.open(img_path) as img:
            # Convert to grayscale if not already
            img = img.convert('L')
            
            # Resize to 28x28 if needed
            if img.size != (28, 28):
                img = img.resize((28, 28))
            
            # Apply transform (this handles conversion to tensor and normalization)
            img_tensor = transform(img)
            
            images.append(img_tensor)
    
    # Stack all images into a single tensor
    images_tensor = torch.stack(images)
    print("######", images_tensor.shape)
    return images_tensor

def labelize(images):
    desc = "Labelizing"
    num_batches = int(np.ceil(images.size(0) / batch_size))
    images = images.reshape(-1, 28, 28).unsqueeze(1)#.repeat(1, 3, 1, 1)
    #resize = partial(F.interpolate, size=(224, 224))

    confidences = []
    labels = []
    for bi in trange(num_batches, desc=desc):
        start = bi * batch_size
        end = start + batch_size
        batch = images[start:end].float()
        confidence,label = torch.exp(vgg5(batch.to(device))).max(dim=1)
        confidences.append(confidence.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    return np.concatenate(labels, axis=0),np.concatenate(confidences, axis=0)


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

X = train_dataset.data[:10000]#.reshape((-1,28*28))[:10000]
y = train_dataset.targets[:10000]


X = load_bw_images("samples")
print(X.shape)
y,confidence = labelize(X)

# Test original image space
tsne_pipeline(X.flatten(start_dim = 1),y, "MNIST Dataset - Original Image Space")

new_X = vgg16_encoding(X)
tsne_pipeline(new_X,y, "MNIST Dataset - VGG 16 (pretrained) Space")

new_X = vgg5_encoding(X)
tsne_pipeline(new_X,y, "MNIST Dataset -  VGG 5 (fine-tuned) Space")

"""     
# Test different embeddings
from sklearn.decomposition import PCA
X_pca = PCA(n_components=20).fit_transform(X)
tsne_pipeline(X_pca,y, "MNIST Dataset - PCA Embeddings")

# You can now try other embedding methods, such as:
# - Autoencoders
# - Word2Vec
# - UMAP
# - etc.
# and pass them to the tsne_pipeline function to visualize the results
"""