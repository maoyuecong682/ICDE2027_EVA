import os
import torch
import numpy as np
from collections import defaultdict

from sampling.embed import load_embed, compute_embed
from sampling.mean_estimate import mean_estimate
from sampling.algos import pruning_functions

# ============================================
def run_sampling_sgd(
        sample,
        label,
        model,
        subset_size : int,
):
    """
    Args:
        sample:
        label:
        model:
        subset_size:

    Returns:

    """
    # 1. Compute Embeddings <Early Exit>

    # 2. Solve Moment Matching Problem
    # 3. return the selected indices

# ============================================
def run_coreset_select(
        dataset,
        embedding_file: str,
        coreset_method: str,
        coreset_size: int,
        embed_bs: int = 512
):
    # --------------------------------------------
    # 1. CLIP Embeddings
    # --------------------------------------------
    print(f"Dataset size: {len(dataset)}")

    # Only compute embeddings if they don't already exist
    if not os.path.exists(embedding_file):
        print("Computing embeddings...")
        compute_embed(
            dataset=dataset,
            batch_size=embed_bs,
            embeddings_file=embedding_file,
        )
        # Embeddings are now saved to disk and can be loaded when needed
    else:
        print("Embeddings already exist, skipping computation.")

    print(f"Loading embeddings from {embedding_file}")
    embeddings = load_embed(embeddings_path=embedding_file)  # memory efficient loading

    # Group by class
    class_to_indices = defaultdict(list)
    for idx, (sample, class_label) in enumerate(dataset):
        # class_label = sample['target']  # assumes dataset[idx]['label'] is the class
        class_to_indices[class_label].append(idx)

    num_classes = len(class_to_indices)
    coreset_size_per_class = coreset_size // num_classes

    selected_indices = []

    for class_label, indices in class_to_indices.items():
        print(f"Coreset Selection for class {class_label} with {len(indices)} samples")

        class_embeddings = embeddings[indices]

        if coreset_method == "gm_matching":
            moment = mean_estimate(
                data=class_embeddings.numpy(),
                estimator="geo_med",
                eps=1e-5,
                max_iter=5000
            )
            moment = torch.from_numpy(moment)
        else:
            moment = torch.mean(class_embeddings, dim=0)

        to_select = int(min(coreset_size_per_class, len(class_embeddings)))
        to_select = max(to_select, 1)  # Ensure at least one sample is selected

        batch_indices = pruning_functions[coreset_method](
            phi=class_embeddings,
            coreset_size=to_select,
            moment=moment,
        )

        selected_indices.extend([indices[i] for i in batch_indices])

    selected_indices = torch.tensor(selected_indices)
    print(f"Total selected: {len(selected_indices)}")
    return selected_indices
# ============================================
#
#     # --------------------------------------------
#     # 2. Calculate Moments
#     # --------------------------------------------
#     if coreset_method == "gm_matching":
#         print(f"Using GM as the center for {coreset_method}")
#         moment = mean_estimate(
#             data=embeddings.numpy(),
#             estimator="geo_med",
#             eps=1e-5,
#             max_iter=5000
#         )
#         moment = torch.from_numpy(moment)
#     else:
#         print(f"Using Empirical Mean as the center for {coreset_method}")
#         moment = torch.mean(embeddings, dim=0)
#
#     # --------------------------------------------
#     # 3. Sampling Algo
#     # --------------------------------------------
#     print("\nPerforming sample selection...")
#
#     # Parameters for batch processing
#     batch_size = coreset_batch_size  # adjust based on your memory constraints
#     print(f"Selecting {coreset_size} samples using {coreset_method} method")
#
#     selected_indices = []
#
#     for i in range(0, len(embeddings), batch_size):
#         batch_embeddings = embeddings[i:i+batch_size]
#         # Calculate batch size proportional to total desired samples
#         to_select = int(np.ceil(coreset_size * (len(batch_embeddings) / len(embeddings))))
#         batch_select_size = min(
#             to_select,
#             len(batch_embeddings)
#         )
#
#         # Select indices for this batch
#         batch_indices = pruning_functions[coreset_method](
#             phi=batch_embeddings,
#             coreset_size=batch_select_size,
#             moment=moment,
#         )
#
#         # Adjust indices to global index space
#         global_indices = batch_indices + i
#         selected_indices.extend(global_indices.tolist())
#
#     selected_indices = torch.tensor(selected_indices)
#     print(f"Selected {len(selected_indices)} samples using {coreset_method} method")
#
#     return selected_indices
