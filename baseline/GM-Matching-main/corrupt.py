import numpy as np

def add_label_noise(dataset, noise_ratio=0.2, num_classes=10):
    targets = np.array(dataset.targets)  # assumes CIFAR-10 style datasets
    n_samples = len(targets)
    n_noisy = int(noise_ratio * n_samples)

    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    for idx in noisy_indices:
        true_label = targets[idx]
        new_label = np.random.choice([i for i in range(num_classes) if i != true_label])
        targets[idx] = new_label

    dataset.targets = targets.tolist()
    return dataset