import numpy as np
from sklearn.decomposition import PCA

# build ssm
def build_ssm(shapes, variance_threshold=0.95):
    """
    Builds a Statistical Shape Model (SSM) using PCA and retains dominant modes.

    Parameters:
    - shapes (numpy array): Shape (n_shapes, n_points, 2), 2D landmark-based shapes.
    - variance_threshold (float): The required variance retention (default: 95%).

    Returns:
    - mean_shape (numpy array): The mean shape (flattened).
    - U_reduced (numpy array): Eigenvectors (shape modes) as columns.
    - eigenvalues (numpy array): Corresponding eigenvalues for retained modes.
    - num_modes (int): Number of modes retained.
    """
    num_shapes, num_points, dim = shapes.shape
    # Step 1: Flatten the shapes into vectors (each shape is a row)
    flattened_shapes = shapes.reshape(num_shapes, num_points * dim)  # Shape: (n_shapes, n_features)
    # Step 2: Compute Mean Shape
    mean_shape = np.mean(flattened_shapes, axis=0)  # Shape: (n_features,)
    centered_shapes = flattened_shapes - mean_shape
    # Step 3: Perform PCA
    pca = PCA()
    pca.fit(centered_shapes)  # Center data before PCA
    # Get the principal components (shape modes)
    U = pca.components_  # These are the shape modes (directions of variation)
    explained_variance = pca.explained_variance_ratio_
    # Step 4: Compute Cumulative Variance
    cumulative_variance = np.cumsum(explained_variance)
    num_modes = np.argmax(cumulative_variance >= variance_threshold) + 1  # Retain modes above threshold
    # Step 5: Extract the reduced U matrix (Eigenvectors as columns) and eigenvalues
    U_reduced = U[0:num_modes].T  # Shape: (n_features, num_modes)
    eigenvalues = pca.explained_variance_[:num_modes]  # Corresponding eigenvalues

    return mean_shape, U_reduced, eigenvalues, num_modes