import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# 原版在模块顶层 `import clip`，但 clip 只服务于图像嵌入。
# 表格数据（adult）直接用原始特征作嵌入并预写 embeddings.pt，走不到 CLIP 分支，
# 因此改为函数内延迟导入，避免没有 clip 包时模块导入直接失败。


def get_embeddings(
    images: Union[torch.Tensor, List[Image.Image]],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Get CLIP embeddings for a batch of images.
    
    Args:
        images: Either a batch of torch tensors [B, C, H, W] normalized to [-1, 1]
               or a list of PIL images
        device: Device to run the model on
            
    Returns:
        torch.Tensor: Tensor of embeddings for the batch [B, embedding_dim]
        @param device:
        @param images:
        @param preprocess:
        @param model:
    """
    import clip  # 延迟导入，只有真正需要 CLIP 时才会失败

    model, preprocess = clip.load("ViT-B/32", device=device)
    processed_images = []
    with torch.no_grad():
        for image in images:
            # Convert to PIL if needed
            if isinstance(image, torch.Tensor):
                # Assuming image is normalized to [-1, 1], convert to [0, 1]
                image = (image + 1) / 2
                image = torch.clamp(image, 0, 1)
                # Convert to PIL
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(224, 224))
                image = image.squeeze(0)
                
                # Handle grayscale images by repeating channels
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                
                image = image.permute(1, 2, 0)
                image = (image * 255).cpu().numpy().astype(np.uint8)
                image = Image.fromarray(image)

            # Preprocess image
            if preprocess:
                image = preprocess(image)
            else:
                # Default CLIP preprocessing if none provided
                image = model.preprocess(image)
            processed_images.append(image)

        # Stack all processed images into a batch
        image_batch = torch.stack(processed_images).to(device)
        embeddings = model.encode_image(image_batch).to("cpu")

    return embeddings


def compute_embed(
    dataset,
    batch_size: int = 32,
    embeddings_file: str = "embeddings.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Get CLIP embeddings for a dataset in batches and save them incrementally.
    
    Args:
        embeddings_file:
        dataset: A PyTorch dataset that returns images
        batch_size: Batch size for processing
        device: Device to run the model on
    """
    # Create dataloader -- keep the indices
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    
    # Get first batch to determine embedding dimension
    print(f"Processing first batch of {batch_size} images...")
    first_batch = next(iter(dataloader))
    if isinstance(first_batch, (tuple, list)):
        images = first_batch[0]
    elif isinstance(first_batch, dict):
        images = first_batch.get("pixel_values", first_batch.get("images"))
    else:
        images = first_batch
    
    first_embeddings = get_embeddings(
        images,
        device=device
    )
    embedding_dim = first_embeddings.shape[1]
    
    # Initialize memory-mapped file for embeddings
    total_size = len(dataset)
    memmap_embeddings = np.memmap(
        embeddings_file + ".tmp",
        dtype=np.float32,
        mode='w+',
        shape=(total_size, embedding_dim)
    )
    
    # Save first batch
    memmap_embeddings[:len(first_embeddings)] = first_embeddings.numpy()
    current_idx = len(first_embeddings)
    
    print(f"Processing remaining {total_size - current_idx} images in batches of {batch_size}...")
    
    # Process remaining batches
    for batch in tqdm(list(dataloader)[1:]):  # Skip first batch since we already processed it
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        elif isinstance(batch, dict):
            images = batch.get("pixel_values", batch.get("images"))
        else:
            images = batch
            
        # Get embeddings for batch
        embeddings = get_embeddings(
            images,
            device=device
        )
        
        # Save batch to memmap file
        batch_size = len(embeddings)
        memmap_embeddings[current_idx:current_idx + batch_size] = embeddings.numpy()
        current_idx += batch_size
        
        # Flush to disk periodically
        memmap_embeddings.flush()
    
    # Convert memmap to tensor and save
    print(f"Converting to tensor and saving to {embeddings_file}")
    torch.save(torch.from_numpy(memmap_embeddings[:]), embeddings_file)
    
    # Clean up temporary file
    del memmap_embeddings
    os.remove(embeddings_file + ".tmp")


def load_embed(
    embeddings_path: str,
    indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Load CLIP embeddings from disk, optionally selecting specific indices.
    Uses memory mapping for efficient loading of large files.
    
    Args:
        embeddings_path: Directory containing the embeddings.pt file
        indices: Optional list of indices to load. If None, loads all embeddings
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"No embeddings found at {embeddings_path}")
    
    # Memory map the embeddings file
    embeddings = torch.load(embeddings_path, map_location='cpu', mmap=True)
    
    if indices is not None:
        return embeddings[indices]

    return embeddings


def visualize_embeddings(
    embeddings_path: str,
    dataset=None,
    num_samples: int = 5000,
    indices: Optional[List[int]] = None,
    save_path: str = "./",
    moments: Optional[dict] = None,
) -> None:
    """
    Visualize the manifold using pre-computed CLIP embeddings.
    
    Args:
        embeddings_path: Path to the saved embeddings
        dataset: Optional dataset object to identify corrupt samples
        indices: Optional list of indices to visualize
        num_samples: Number of samples to visualize
        save_path: Path to save visualization plots
        moments: Optional dictionary of moments to plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    sns.set_theme(style="white")
    sns.set_context("poster")
    
    # Load embeddings
    embeddings = load_embed(embeddings_path)
    
    # Sample indices if needed
    if indices is not None:
        embeddings = embeddings[indices]

    else:
        indices = list(range(len(embeddings)))
        if num_samples < len(embeddings):
            indices = np.random.choice(indices, num_samples, replace=False)
            embeddings = embeddings[indices]
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get corrupt/clean sample information if dataset is provided
    corrupt_mask = None
    clean_mask = None
    if dataset is not None and hasattr(dataset, 'corrupt_samples'):
        if isinstance(dataset.corrupt_samples, dict):
            corrupt_samples = set(dataset.corrupt_samples.values())
        else:
            corrupt_samples = dataset.corrupt_samples
            
        # Create masks for the sampled indices
        corrupt_mask = torch.tensor([idx in corrupt_samples for idx in indices])
        clean_mask = ~corrupt_mask
        
        print(f"\nIn visualization subset:")
        print(f"num of corrupt samples: {corrupt_mask.sum().item()}")
        print(f"num of clean samples: {clean_mask.sum().item()}")
    
    # PCA Visualization
    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(embeddings.cpu())
    
    plt.figure(figsize=(10, 8), dpi=300)
    if corrupt_mask is not None:
        sns.scatterplot(x=reduced_embeddings_pca[clean_mask, 0],
                       y=reduced_embeddings_pca[clean_mask, 1],
                       label='clean samples', alpha=0.8, color='gray', s=200, edgecolor='black')

        sns.scatterplot(x=reduced_embeddings_pca[corrupt_mask, 0],
                       y=reduced_embeddings_pca[corrupt_mask, 1],
                       label='corrupt samples', alpha=0.8, color='lightcoral', s=200, edgecolor='black')
    else:
        sns.scatterplot(x=reduced_embeddings_pca[:, 0], y=reduced_embeddings_pca[:, 1],
                       label='samples', alpha=0.8, color='gray', s=200, edgecolor='black')
    
    # Plot moments if provided
    if moments is not None:
        colors = ['green', 'red', 'purple', 'magenta']
        # markers = ['*', 'P', 'D']
        for (name, moment), color in zip(moments.items(), colors):
            moment_reduced = pca.transform(moment.cpu().unsqueeze(0))
            plt.scatter(moment_reduced[:, 0], moment_reduced[:, 1], 
                       label=name, color=color,
                       s=800, edgecolor='black', linewidth=2)

    plt.xlabel('PC 1', fontsize=20, weight='bold')
    plt.ylabel('PC 2', fontsize=20, weight='bold')
    plt.legend(
        frameon=True,
        edgecolor='black',
        framealpha=0.8,
        markerscale=1,
        fancybox=True,
        prop={'weight': 'bold', 'size': 14}
    )
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "pca.pdf"))
    plt.close()
    
    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings_tsne = tsne.fit_transform(embeddings.cpu())
    
    plt.figure(figsize=(10, 8), dpi=300)
    if corrupt_mask is not None:
        sns.scatterplot(x=reduced_embeddings_tsne[clean_mask, 0],
                       y=reduced_embeddings_tsne[clean_mask, 1],
                       label='clean samples', alpha=0.8, color='gray', s=200, edgecolor='black')
        sns.scatterplot(x=reduced_embeddings_tsne[corrupt_mask, 0],
                       y=reduced_embeddings_tsne[corrupt_mask, 1],
                       label='corrupt samples', alpha=0.8, color='lightcoral', s=200, edgecolor='black')
    else:
        sns.scatterplot(x=reduced_embeddings_tsne[:, 0], y=reduced_embeddings_tsne[:, 1],
                       alpha=0.8, color='gray', s=200, edgecolor='black')
    
    # Plot moments if provided
    if moments is not None:
        # Include moments in t-SNE by concatenating with embeddings
        all_points = torch.cat([embeddings] + [m.unsqueeze(0) for m in moments.values()])
        reduced_all = tsne.fit_transform(all_points.cpu())
        
        # Plot moments (last points in the reduced array)
        moment_idx = len(embeddings)
        colors = ['green', 'red', 'purple', 'magenta']
        # markers = ['*', 'P', 'D']
        for (name, _), color in zip(moments.items(), colors):
            plt.scatter(reduced_all[moment_idx, 0], reduced_all[moment_idx, 1],
                       label=name, color=color,
                       s=800, edgecolor='black', linewidth=2)
            moment_idx += 1

    plt.xlabel('t-SNE 1', fontsize=20, weight='bold')
    plt.ylabel('t-SNE 2', fontsize=20, weight='bold')
    plt.legend(
        frameon=True,
        edgecolor='black',
        framealpha=0.8,
        markerscale=1,
        fancybox=True,
        prop={'weight': 'bold', 'size': 14}
    )
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "tsne.pdf"))
    plt.close()

