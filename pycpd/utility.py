import numpy as np
import json
import os
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf
import open3d as o3d



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

def pca_kernel(X, mean_shape, U, eigenvalues):
    """
    Compute a PCA-based kernel matrix for shape deformation.
    """
    n_points_x = X.shape[0]

    # Ensure U is properly shaped
    if len(U.shape) == 3:
        U_resh = U  # Already correctly shaped
    else:
        num_modes = U.size // (n_points_x * 3)  # Compute dynamically
        if num_modes * n_points_x * 3 != U.size:
            raise ValueError(f"U cannot be reshaped correctly. U.shape: {U.shape}, Expected: ({n_points_x}, 3, ?)")
        U_resh = U.reshape(n_points_x, 3, num_modes)

    print(f"Reshaped U: {U_resh.shape}")

    # Mode weights computation
    M = np.einsum('ikm,jkm->ijm', U_resh, U_resh)

    # Ensure eigenvalues are the correct length
    if eigenvalues.shape[0] > M.shape[-1]:
        eigenvalues = eigenvalues[:M.shape[-1]]  # Slice down if too many modes

    print(f"eigenvalues.shape: {eigenvalues.shape}, expected: ({M.shape[-1]},)")

    # Normalize using eigenvalues
    M /= (np.log(eigenvalues + 1e-8) + 1)

    # Compute final kernel matrix
    K = np.sum(M, axis=2)

    # Normalize to [0,1]
    K = K / np.max(K)

    return K


def pca_kernel_new(X, mean_shape, U, eigenvalues, beta=1.0, use_spatial=False):
    n_points_x = X.shape[0]

    # Reshape U
    if len(U.shape) == 3:
        U_resh = U
    else:
        num_modes = U.size // (n_points_x * 3)
        U_resh = U.reshape(n_points_x, 3, num_modes)

    # Mode weights with proper eigenvalue regularization
    M = np.einsum('ikm,jkm->ijm', U_resh, U_resh)

    # Use inverse  for regularization
    # Only use significant modes
    significant_modes = eigenvalues > eigenvalues[0] * 0.05  # 5% threshold
    M[:, :, ~significant_modes] = 0

    # eigenvalue weighting
    weights = np.zeros_like(eigenvalues)
    weights[significant_modes] = 1.0 / np.sqrt(eigenvalues[significant_modes])
    M *= weights[np.newaxis, np.newaxis, :]

    # Sum over modes
    K_pca = np.sum(M, axis=2)

    if use_spatial:
        # Combine with spatial kernel for locality
        distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        K_spatial = np.exp(-distances / (2 * beta ** 2))

        # Combine PCA and spatial information
        K = K_pca * K_spatial
    else:
        K = K_pca

    # Add small diagonal term for numerical stability
    K += np.eye(n_points_x) * 1e-6

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


def plot_metrics_comparison(results):
    """Plots comparison of RMSE and execution time given registration results."""
    # Extract registration method names, rmse values and execution times
    method_names = list(results.keys())
    rmse_values = [results[m]['metrics']['rmse'] for m in method_names]
    times = [results[m]['time'] for m in method_names]

    # Bar chart for RMSE
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(method_names, rmse_values, color='skyblue')
    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")
    plt.xlabel("Method")

    # Bar chart for Execution Time
    plt.subplot(1, 2, 2)
    plt.bar(method_names, times, color='salmon')
    plt.title("Execution Time")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Method")

    plt.tight_layout()
    plt.show()


def plot_registration_comparison(X, Y, results):
    """Show side-by-side plots comparing the point set registration for each method."""

    dim = X.shape[1]
    method_names = list(results.keys())

    num_methods = len(method_names)
    plt.figure(figsize=(6 * num_methods, 5))

    for i, m in enumerate(method_names, 1):
        TY = results[m]['transformed_points']

        plt.subplot(1, num_methods, i)
        if dim == 2:
            # 2D scatter
            plt.scatter(X[:, 0], X[:, 1], c='red', label='Target (X)', s=8)
            plt.scatter(TY[:, 0], TY[:, 1], c='blue', label=f'{m} Registered (Y)', s=8)
            plt.scatter(Y[:, 0], Y[:, 1], c='green', label='Original Y', s=8, alpha=0.3)
            plt.title(f"{m} Registration")
            plt.legend(loc='upper right')
        elif dim == 3:
            # 3D scatter example
            ax = plt.gca(projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='red', label='Target (X)', s=8)
            ax.scatter(TY[:, 0], TY[:, 1], TY[:, 2], c='blue', label=f'{m} Registered (Y)', s=8)
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='green', label='Original Y', s=8, alpha=0.3)
            ax.set_title(f"{m} Registration")
            ax.legend()
        else:
            raise ValueError("dim must be 2 or 3.")

    plt.tight_layout()
    plt.show()

def compute_rmse(A,B):
    return np.sqrt(np.mean(np.sum((A-B)**2,axis=1)))

def frobenius_covariance_distance(A, B):
    """
    Computes the Frobenius norm between two point clouds.
    """
    # Compute covariance matrices (rows = samples, columns = dimensions)
    cov1 = np.cov(A, rowvar=False)
    cov2 = np.cov(B, rowvar=False)

    # Frobenius norm of the difference
    frobenius_distance = np.linalg.norm(cov1 - cov2, ord='fro') # ∥Σ1 - Σ2∥f

    return frobenius_distance


def hausdorff_distance(A_flat, B_flat):
    """
    Computes Hausdorff distance between two flattened point clouds using NumPy only.
    Parameters:
    - A_flat: 1D NumPy array of shape (n_points * point_dim,)
    - B_flat: 1D NumPy array of shape (m_points * point_dim,)
    - point_dim: dimensionality of each point (e.g., 3 for 3D)
    Returns:
    - Symmetric Hausdorff distance
    """
    A = A_flat.reshape(-1, 3)
    B = B_flat.reshape(-1, 3)

    def directed_hausdorff(U, V):
        dists = np.linalg.norm(U[:, np.newaxis, :] - V[np.newaxis, :, :], axis=2)
        return np.max(np.min(dists, axis=1))

    forward = directed_hausdorff(A, B)
    backward = directed_hausdorff(B, A)
    return max(forward, backward)


def calculate_tps_transform(source_points, target_points):
    print(f"Creating TPS transform with {source_points.shape[0]} point pairs...")

    try:
        # Create separate RBF interpolators for each coordinate
        rbf_x = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2],
                    target_points[:, 0], function='thin_plate', smooth=1e-6)
        rbf_y = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2],
                    target_points[:, 1], function='thin_plate', smooth=1e-6)
        rbf_z = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2],
                    target_points[:, 2], function='thin_plate', smooth=1e-6)

        def transform_function(points):
            try:
                x_transformed = rbf_x(points[:, 0], points[:, 1], points[:, 2])
                y_transformed = rbf_y(points[:, 0], points[:, 1], points[:, 2])
                z_transformed = rbf_z(points[:, 0], points[:, 1], points[:, 2])
                return np.vstack([x_transformed, y_transformed, z_transformed]).T
            except Exception as e:
                print(f"TPS transform evaluation failed: {e}")
                return points.copy()

        return transform_function

    except Exception as e:
        print(f"TPS transform creation failed: {e}")

        def identity_transform(points):
            return points.copy()

        return identity_transform


def downsample_point_cloud(points, target_voxel_size=0.505):
    """Downsample a point cloud using Open3D's voxel downsampling method."""
    try:
        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Perform voxel downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=target_voxel_size)

        # Convert back to numpy array
        downsampled_points = np.asarray(downsampled_pcd.points)

        print(f"Downsampling: {len(pcd.points)} → {len(downsampled_pcd.points)} points")
        return downsampled_points

    except Exception as e:
        print(f"Downsampling failed: {e}, using original points")
        return points






