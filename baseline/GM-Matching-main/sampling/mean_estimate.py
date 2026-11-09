"""
Robust Estimation of Location Parameters
"""
import numpy as np
from scipy.spatial.distance import cdist, euclidean


def mean_estimate(
        data: np.ndarray,
        estimator: str = 'geo_med',
        algo: str = 'weiszfeld',
        eps: float = 1e-5,
        max_iter: int = 5000
) -> np.ndarray:
    """
    Robust Estimation of Location Parameters
    :param data: np.ndarray of shape (n_samples, n_features)
    :param estimator: str, 'geo_med' for geometric median
    :param algo: str, 'weiszfeld' or 'vardi' for geometric median
    :param eps: float, stopping criteria
    :param max_iter: int, maximum number of iterations

    returns np.ndarray of shape (n_features, 1) - robust estimate of location parameter (mean)
    """
    if estimator == 'geo_med':
        if algo == 'weiszfeld':
            return weiszfeld_gm(
                data=data,
                eps=eps,
                max_iter=max_iter
            )
        elif algo == 'vardi':
            return vardi_gm(
                data=data,
                eps=eps,
                max_iter=max_iter
            )
        elif algo == 'cohen':
            return cohen_geometric_median(
                points=data,
                epsilon=eps,
                max_iter=max_iter
            )
        else:
            raise NotImplementedError

    elif estimator == "co_med":
        return np.median(data, axis=0)
    else:
        raise NotImplementedError


def cohen_geometric_median(points, epsilon=1e-5, max_iter=100):
    """
    Approximates the geometric median using an iterative refinement method inspired by Cohen et al.

    Args:
        points (np.ndarray): shape (n, d) array of n points in d-dimensional space.
        epsilon (float): convergence threshold for relative improvement.
        max_iter (int): maximum number of iterations.

    Returns:
        np.ndarray: approximate geometric median.
    """
    n, d = points.shape

    # Step 1: Initialize geometric median with mean
    x = np.mean(points, axis=0)

    for iter in range(max_iter):
        # Step 2: Compute distances and weights
        distances = np.linalg.norm(points - x, axis=1)
        total_distance = np.sum(distances)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-12)

        weights = 1 / distances
        weights /= np.sum(weights)  # Normalize to sum to 1

        # Step 3: Refine median using weighted average
        new_x = np.sum(weights[:, None] * points, axis=0)

        # Step 4: Check for convergence
        relative_improvement = np.linalg.norm(new_x - x) / (np.linalg.norm(x) + 1e-12)
        if relative_improvement < epsilon:
            break

        x = new_x

    return x


def weiszfeld_gm(
        data: np.ndarray,
        eps: float = 1e-5,
        max_iter: int = 1000
):
    # inspired by: https://github.com/mrwojo
    """
    Implements:
    On the point for which the sum of the distances to n given points is minimum (1927)
    E Weiszfeld, F Plastria; Annals of Operations Research
    """
    # initial Guess : centroid / empirical mean
    mu = np.mean(a=data, axis=0)
    num_iter = 0
    while num_iter < max_iter:
        # noinspection PyTypeChecker
        distances = cdist(data, [mu]).astype(mu.dtype)
        distances = np.where(distances == 0, 1e-8, distances)
        mu1 = (data / distances).sum(axis=0) / (1. / distances).sum(axis=0)
        guess_movement = np.sqrt(((mu - mu1) ** 2).sum())
        mu = mu1
        if guess_movement <= eps:
            return mu
        num_iter += 1
    # print('Ran out of Max iter for GM - returning sub optimal answer')
    return mu


def vardi_gm(
        data: np.ndarray,
        eps: float = 1e-5,
        max_iter: int = 1000
) -> np.ndarray:
    # Copyright (c) Orson Peters
    # Licensed under zlib License
    # Reference: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
    """
    Implementation of "The multivariate L1-median and associated data depth;
    Yehuda Vardi and Cun-Hui Zhang; PNAS'2000"
    """
    # initial guess
    mu = np.mean(a=data, axis=0)
    mu = np.nan_to_num(mu, copy=False, nan=0, posinf=0, neginf=0)
    num_iter = 1
    while num_iter < max_iter:
        # noinspection PyTypeChecker
        D = cdist(data, [mu]).astype(mu.dtype)
        non_zeros = (D != 0)[:, 0]
        D_inv = 1 / D[non_zeros]
        W = np.divide(D_inv, sum(D_inv))
        T = np.sum(W * data[non_zeros], 0)
        num_zeros = len(data) - np.sum(non_zeros)

        if num_zeros == 0:
            mu1 = T
        elif num_zeros == len(data):
            return mu
        else:
            r = np.linalg.norm((T - mu) * sum(D_inv))
            r_inv = 0 if r == 0 else num_zeros / r
            mu1 = max(0, 1 - r_inv) * T + min(1, r_inv) * mu

        mu1 = np.nan_to_num(mu1, copy=False, nan=0, posinf=0, neginf=0)

        if euclidean(mu, mu1) < eps:
            return mu

        mu = mu1
        num_iter += 1

    print('Ran out of Max iter for GM -')
    return mu