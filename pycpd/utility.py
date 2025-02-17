import numpy as np
import json
import os


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))


def pca_kernel(X, mean_shape, U, eigenvalues, beta=None, Y=None):
    """
    Compute a PCA-based kernel matrix for shape deformation.
    """
    if Y is None:
        Y = X

    n_points_x = X.shape[0]
    n_points_y = Y.shape[0]

    # Initialize kernel matrix with correct dimensions
    K = np.zeros((n_points_x, n_points_y))

    # Scale factor for distances
    scale = np.mean(np.sum((X - Y[0]) ** 2, axis=1))
    if scale == 0:
        scale = 1.0

    # Compute pairwise distances
    for i in range(n_points_x):
        for j in range(n_points_y):
            # Compute normalized distance between points
            dist = np.sum((X[i] - Y[j]) ** 2) / scale

            # Weight by PCA modes
            mode_weights = 0
            for k in range(len(eigenvalues)):
                mode_contribution = (U[i * 3:(i + 1) * 3, k].dot(U[j * 3:(j + 1) * 3, k]))
                mode_weights += mode_contribution / (eigenvalues[k] + 1e-8)

            # Combine distance and mode weights
            K[i, j] = np.exp(-dist / 2) * (1 + mode_weights)

    # Print debug info about kernel
    print(f"Kernel stats - Min: {np.min(K):.6f}, Max: {np.max(K):.6f}, Mean: {np.mean(K):.6f}")

    # Normalize kernel to [0, 1] range
    K = (K - np.min(K)) / (np.max(K) - np.min(K) + 1e-8)

    return K


def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


def get_slicer_positions_txt(json_file_path):
    """
    Reads a Slicer Markups JSON file from a given path, extracts position data,
    and returns it as a formatted string in scientific notation.

    :param json_file_path: Path to the JSON file.
    :return: A formatted string of positions or an error message if the file is missing.
    """
    if not os.path.exists(json_file_path):
        return f"Error: File not found at {json_file_path}"

    # Read and parse the JSON file
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Extract control points from the first markup entry
    control_points = json_data.get("markups", [])[0].get("controlPoints", [])

    # Format positions
    formatted_positions = "\n".join(
        " ".join(f"{coord:.18e}" for coord in entry["position"]) for entry in control_points
    )

    return formatted_positions

def calculate_registration_metrics(source, target):
    """Calculate alignment metrics between source and target point clouds."""
    diff = source - target
    return {
        'rmse': np.sqrt(np.mean(np.sum(diff**2, axis=1))),
        'mae': np.mean(np.abs(diff)),
        'max_error': np.max(np.abs(diff)),
        'rmse_per_axis': np.sqrt(np.mean(diff**2, axis=0))
    }



