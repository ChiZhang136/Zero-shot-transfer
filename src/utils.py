import os
import numpy as np


def similarity_weights(gammas, eps=1e-6, power=1.0):
    """
    Similarity-aware weights:

        w_k proportional to 1 / (Gamma_k + eps)^power.

    Smaller Gamma_k receives a larger weight.
    """
    gammas = np.asarray(gammas, dtype=float)

    scores = 1.0 / np.power(gammas + eps, power)

    return scores / np.sum(scores)


def uniform_weights(K):
    """
    Uniform averaging weights.
    """
    return np.ones(K) / K


def mean_and_sem(values, axis=0):
    """
    Compute mean and standard error of the mean.
    """
    values = np.asarray(values)

    mean = np.mean(values, axis=axis)
    sem = np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])

    return mean, sem


def ensure_dir(path):
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)