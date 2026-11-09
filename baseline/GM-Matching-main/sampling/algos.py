"""
(Combinatorial) Data Pruning over Embeddings.
"""
import torch
from sampling.mean_estimate import mean_estimate


def compute_center(target_moment: str, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the Center of the Class
    --------------------------------
    :param target_moment: str, 'mean' / 'geo_med' / 'co_med'
    :param phi: torch.Tensor of size N x D (N samples in D-dimensional space)
    """
    # print("Computing Center of the Class using {}".format(target_moment))
    if target_moment == 'mean':
        class_center = torch.mean(phi, dim=0)

    elif target_moment == 'geo_med':
        class_center = mean_estimate(
            data=phi.cpu().numpy(),
            estimator='geo_med',
            eps=1e-5,
            max_iter=1000
        )
        class_center = torch.tensor(class_center, dtype=phi.dtype, device=phi.device)

    else:
        raise NotImplementedError

    return class_center


# ===============================================================
# Geometric Pruning Strategies
# ===============================================================
def moderate(
        phi: torch.Tensor,
        coreset_size: int,
        moment: torch.Tensor = None,
        target_moment='mean',  # mean, geo_med, co_med
        **kwargs
) -> torch.Tensor:
    """

    Moderate Coreset Sampling
    -------------------------
    Selects samples around the median distance from the class center.
    Reference:
    https://openreview.net/pdf?id=7D5EECbOaf9
    MODERATE CORESET: A UNIVERSAL METHOD OF DATA SELECTION FOR REAL-WORLD DATA-EFFICIENT DEEP LEARNING; ICLR 2023

    @param phi:
    torch.Tensor of size N x D from p(x | y = j)

    @param coreset_size: int
    size of the pruned dataset (coreset)

    @param moment: torch.Tensor

    @param target_moment: str

    @return selected_indices:
    torch.Tensor list of indices of selected samples

    """

    if moment is None:
        # compute the center of the class [Eq. 1]
        class_center = compute_center(target_moment, phi)
    else:
        class_center = moment
    # select samples around the sample with median distance from the center [Eq. 2]
    distances = torch.norm(phi - class_center, dim=1)
    low_idx = int(round(0.5 * distances.shape[0] - coreset_size / 2))
    high_idx = int(low_idx + coreset_size) #round(0.5 * distances.shape[0] + coreset_size / 2)
    sorted_idx = torch.argsort(distances)
    selected_indices = sorted_idx[low_idx:high_idx]

    return selected_indices


def hard(
        phi: torch.Tensor,
        coreset_size: int,
        moment: torch.Tensor = None,
        target_moment='mean',  # mean, geo_med, co_med
        **kwargs
) -> torch.Tensor:
    """
    Hard Samples:
    Selects the hardest examples from the dataset i.e. farthest from the class centers E(x | y=j) for-all j \in Y

    ref:
    Beyond neural scaling laws: beating power law scaling via data pruning. NeuRips 2022.
    https://openreview.net/forum?id=UmvSlP-PyV

    @param phi:
    torch.Tensor of size N x D

    @param coreset_size: int
    size of the pruned dataset (coreset)

    @param moment: torch.Tensor

    @param target_moment: str
    how to compute center (mean, geo_med, co_med)

    @return hem_indices:
    torch.Tensor list of indices of selected samples
    """

    if moment is None:
        class_center = compute_center(target_moment, phi)
    else:
        class_center = moment
    distances = torch.norm(phi - class_center, dim=1)
    hem_indices = torch.argsort(distances, descending=True)[:coreset_size]

    return hem_indices


def easy(
        phi: torch.Tensor,
        coreset_size: int,
        moment: torch.Tensor = None,
        target_moment='mean',
        **kwargs
) -> torch.Tensor:
    """
    Easy Examples
    --------------------------------
    Selects the easiest examples from the dataset i.e. closest to the class centers E(x | y=j) for-all j \in Y

    @param phi:
    torch.Tensor of size N x D

    @param coreset_size: int
    size of the pruned dataset (coreset)

    @param moment: torch.Tensor

    @param target_moment: str
    how to compute center (mean, geo_med, co_med)

    @return eem_indices:
    torch.Tensor list of indices of selected samples
    """

    # Get the center of the class
    if moment is None:
        class_center = compute_center(target_moment, phi)
    else:
        class_center = moment
    distances = torch.norm(phi - class_center, dim=1)
    eem_indices = torch.argsort(distances, descending=False)[:coreset_size]

    return eem_indices.cpu()

def random(
        phi: torch.Tensor,
        coreset_size: int,
        **kwargs
) -> torch.Tensor:
    """
    Random Sampling
    --------------------------------
    @param phi:
    torch.Tensor of size N x D

    @param coreset_size: int
    size of the pruned dataset (coreset)

    @return rnd_indices:
    torch.Tensor list of selected samples

    """

    # Random Sampling
    rnd_indices = torch.randperm(phi.size(0))[:coreset_size]

    return rnd_indices.cpu()


# ===============================================================
# Herding
# ===============================================================

def herding(
        phi: torch.Tensor,
        coreset_size: int,
        dist_measure: str = 'cosine',  # L2, cosine
        init: str = 'random',  # mean , random , target_moment
        moment: torch.Tensor = None,
        target_moment: str = 'mean',  # mean, geo_med, co_med
        **kwargs
) -> torch.Tensor:
    """

    Herding on Hypersphere with PyTorch :
    ==========================================

    @param phi:
    torch.Tensor of size N x D

    @param coreset_size: int
    size of the pruned dataset (coreset)

    @param dist_measure: str
    Distance Measure to use for Herding (L2, cosine)

    @param init: str
    Initialization for Herding (mean, random, target_moment)

    # @param n_init: int
    # number of random initializations

    @param moment: torch.Tensor

    @param target_moment: str
    robustness parameter for pruning algorithms. (mean, geo_med, co_med)

    """

    # Set the Moment Matching Objective for Herding
    if moment is None:
        class_center = compute_center(target_moment=target_moment, phi=phi)
    else:
        class_center = moment
    # Normalize if cosine distance is used
    if dist_measure == 'cosine':
        phi = torch.nn.functional.normalize(phi, p=2, dim=1)
        class_center = torch.nn.functional.normalize(class_center, p=2, dim=0)

    # initialize the direction to explore
    kh_indices = []

    if init == 'mean':
        w_t = torch.mean(phi, dim=0)

    elif init == 'random':
        w_t = phi[torch.randperm(phi.size(0))[:1]][0]

    elif init == 'target_moment':
        w_t = class_center

    else:
        raise NotImplementedError

    while len(kh_indices) < coreset_size:
        # compute scores
        if dist_measure == 'cosine':
            scores = torch.matmul(phi, w_t)
            indices = scores.argsort(descending=True)

        elif dist_measure == 'L2':
            scores = torch.norm(phi - w_t, dim=1)
            indices = scores.argsort(descending=False)

        else:
            raise NotImplementedError

        # perform updates
        new_ind = next((idx.item() for idx in indices if idx.item() not in kh_indices), None)
        w_t += class_center - phi[new_ind]

        kh_indices.append(new_ind)

    kh_indices = torch.tensor(kh_indices, dtype=torch.long)

    return kh_indices.cpu()


def geo_med_matching(
        phi: torch.Tensor,
        coreset_size: int,
        dist_measure: str = 'cosine',
        moment: torch.Tensor = None,
        init: str = 'random',
        **kwargs
) -> torch.Tensor:
    """
    Robust Herding on Hypersphere with PyTorch
    """
    return herding(
        phi=phi,
        coreset_size=coreset_size,
        dist_measure=dist_measure,
        init=init,
        moment=moment,
        target_moment='geo_med',
        **kwargs
    )

# ===============================================================
pruning_functions = {
			"random": random,
			"easy": easy,
			"hard": hard,
			"moderate": moderate,
			"herding": herding,
			"gm_matching": geo_med_matching,
		}