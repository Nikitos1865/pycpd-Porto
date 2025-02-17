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

    Parameters
    ----------
    X: numpy array (n_points_x, n_dimensions)
        The source point cloud
    mean_shape: numpy array (n_features,)
        The mean shape from SSM
    U: numpy array (n_features, n_modes)
        The reduced eigenvector matrix (shape modes)
    eigenvalues: numpy array (n_modes,)
        The eigenvalues corresponding to retained modes
    beta: float, optional
        Scale parameter (not used in PCA kernel but kept for compatibility)
    Y: numpy array, optional (n_points_y, n_dimensions)
        The target point cloud. If None, Y = X is used
    """
    if Y is None:
        Y = X

    n_points_x = X.shape[0]
    n_points_y = Y.shape[0]
    D = X.shape[1]  # dimensionality (3 for 3D points)

    # Reshape X and Y to match the shape of mean_shape and U
    X_flat = X.reshape(1, -1)  # (1, n_points * D)
    Y_flat = Y.reshape(1, -1)  # (1, n_points * D)

    # Center the points
    X_centered = X_flat - mean_shape
    Y_centered = Y_flat - mean_shape

    # Project onto shape space
    X_proj = np.dot(X_centered, U)  # (1, n_modes)
    Y_proj = np.dot(Y_centered, U)  # (1, n_modes)

    # Initialize kernel matrix
    K = np.zeros((n_points_x, n_points_y))

    # Compute kernel values for each pair of points
    for i in range(n_points_x):
        for j in range(n_points_y):
            # Extract corresponding projections for each point
            x_point_proj = X_proj[:, :]  # Use full projection
            y_point_proj = Y_proj[:, :]  # Use full projection

            # Compute weighted similarity in shape space
            K[i, j] = np.sum((x_point_proj * y_point_proj) / np.sqrt(eigenvalues + 1e-8))

    # Normalize kernel
    K = K / (np.max(np.abs(K)) + 1e-8)
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



