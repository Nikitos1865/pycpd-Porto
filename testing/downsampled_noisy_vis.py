import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.pca_registration_v2 import PCADeformableRegistration2
from pycpd.deformable_registration import DeformableRegistration
from pycpd.ssm import build_ssm
from scipy.interpolate import Rbf
from scipy.spatial.transform import Rotation
from scipy.linalg import svd as scipy_svd
import pandas as pd
from pathlib import Path
import open3d as o3d


def add_noise_to_point_cloud(points, rotation_std=15.0, scale_std=0.1, translation_std=10.0, seed=None):
    """Add controlled noise to point cloud for testing robustness."""
    if seed is not None:
        np.random.seed(seed)

    # Copy points to avoid modifying original
    noisy_points = points.copy()

    # Random rotation (convert degrees to radians)
    rotation_angles = np.random.normal(0, np.radians(rotation_std), 3)
    rotation = Rotation.from_euler('xyz', rotation_angles)
    rotation_matrix = rotation.as_matrix()

    # Random scaling (ensure positive scaling)
    scale_factor = np.abs(np.random.normal(1.0, scale_std))

    # Random translation
    translation = np.random.normal(0, translation_std, 3)

    # Apply transformations: scale -> rotate -> translate
    noisy_points = noisy_points * scale_factor
    noisy_points = noisy_points @ rotation_matrix.T
    noisy_points = noisy_points + translation

    # Store transformation info for logging
    transform_info = {
        'rotation_angles_deg': np.degrees(rotation_angles),
        'scale_factor': scale_factor,
        'translation': translation,
        'rotation_matrix': rotation_matrix
    }

    return noisy_points, transform_info


def visualize_transformation(source_points, target_points, transformed_points, title_prefix, save_dir,
                             source_label="Source", target_label="Target", transformed_label="Transformed",
                             subsample=None):
    """
    Visualize the transformation from source to target, showing displacement vectors.

    Parameters:
    -----------
    source_points : np.array (N, 3)
        Original source points
    target_points : np.array (N, 3)
        Target points that source should align to
    transformed_points : np.array (N, 3)
        Points after transformation is applied
    title_prefix : str
        Prefix for plot titles
    save_dir : str
        Directory to save plots
    subsample : int, optional
        If provided, randomly subsample this many points for clearer visualization
    """

    # Subsample points if requested (for clearer visualization with large point clouds)
    if subsample and len(source_points) > subsample:
        indices = np.random.choice(len(source_points), subsample, replace=False)
        source_viz = source_points[indices]
        target_viz = target_points[indices] if target_points.shape[0] == source_points.shape[0] else target_points
        transformed_viz = transformed_points[indices] if transformed_points.shape[0] == source_points.shape[
            0] else transformed_points
    else:
        source_viz = source_points
        target_viz = target_points
        transformed_viz = transformed_points

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Source vs Target
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(source_viz[:, 0], source_viz[:, 1], source_viz[:, 2],
                c='red', marker='o', s=20, alpha=0.6, label=source_label)
    if target_viz.shape[0] == source_viz.shape[0]:
        ax1.scatter(target_viz[:, 0], target_viz[:, 1], target_viz[:, 2],
                    c='blue', marker='^', s=20, alpha=0.6, label=target_label)
    ax1.set_title(f'{title_prefix}: {source_label} vs {target_label}')
    ax1.legend()
    ax1.set_xlabel('X');
    ax1.set_ylabel('Y');
    ax1.set_zlabel('Z')

    # Plot 2: Source vs Transformed
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(source_viz[:, 0], source_viz[:, 1], source_viz[:, 2],
                c='red', marker='o', s=20, alpha=0.6, label=source_label)
    ax2.scatter(transformed_viz[:, 0], transformed_viz[:, 1], transformed_viz[:, 2],
                c='green', marker='s', s=20, alpha=0.6, label=transformed_label)
    ax2.set_title(f'{title_prefix}: {source_label} vs {transformed_label}')
    ax2.legend()
    ax2.set_xlabel('X');
    ax2.set_ylabel('Y');
    ax2.set_zlabel('Z')

    # Plot 3: Target vs Transformed
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    if target_viz.shape[0] == transformed_viz.shape[0]:
        ax3.scatter(target_viz[:, 0], target_viz[:, 1], target_viz[:, 2],
                    c='blue', marker='^', s=20, alpha=0.6, label=target_label)
        ax3.scatter(transformed_viz[:, 0], transformed_viz[:, 1], transformed_viz[:, 2],
                    c='green', marker='s', s=20, alpha=0.6, label=transformed_label)
        ax3.set_title(f'{title_prefix}: {target_label} vs {transformed_label}')
    else:
        ax3.scatter(transformed_viz[:, 0], transformed_viz[:, 1], transformed_viz[:, 2],
                    c='green', marker='s', s=20, alpha=0.6, label=transformed_label)
        ax3.set_title(f'{title_prefix}: {transformed_label} only')
    ax3.legend()
    ax3.set_xlabel('X');
    ax3.set_ylabel('Y');
    ax3.set_zlabel('Z')

    # Plot 4: Displacement vectors (if same number of points)
    if source_viz.shape[0] == transformed_viz.shape[0]:
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')

        # Calculate displacements
        displacements = transformed_viz - source_viz
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)

        # Plot displacement vectors (subsample for clarity)
        step = max(1, len(source_viz) // 50)  # Show at most 50 vectors
        for i in range(0, len(source_viz), step):
            ax4.quiver(source_viz[i, 0], source_viz[i, 1], source_viz[i, 2],
                       displacements[i, 0], displacements[i, 1], displacements[i, 2],
                       color='purple', alpha=0.7, arrow_length_ratio=0.1)

        ax4.scatter(source_viz[::step, 0], source_viz[::step, 1], source_viz[::step, 2],
                    c='red', marker='o', s=30, alpha=0.8)
        ax4.set_title(
            f'{title_prefix}: Displacement Vectors\nMax: {displacement_magnitudes.max():.3f}, Mean: {displacement_magnitudes.mean():.3f}')
        ax4.set_xlabel('X');
        ax4.set_ylabel('Y');
        ax4.set_zlabel('Z')

    # Plot 5: Displacement magnitude histogram
    if source_viz.shape[0] == transformed_viz.shape[0]:
        ax5 = fig.add_subplot(2, 3, 5)
        displacements = transformed_viz - source_viz
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)

        ax5.hist(displacement_magnitudes, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_xlabel('Displacement Magnitude')
        ax5.set_ylabel('Frequency')
        ax5.set_title(
            f'{title_prefix}: Displacement Distribution\nMean: {displacement_magnitudes.mean():.3f} ± {displacement_magnitudes.std():.3f}')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Error analysis (if target available and same size)
    if target_viz.shape[0] == transformed_viz.shape[0]:
        ax6 = fig.add_subplot(2, 3, 6)

        # Calculate errors
        initial_errors = np.linalg.norm(source_viz - target_viz, axis=1)
        final_errors = np.linalg.norm(transformed_viz - target_viz, axis=1)

        ax6.scatter(initial_errors, final_errors, alpha=0.6, s=20)
        ax6.plot([0, max(initial_errors.max(), final_errors.max())],
                 [0, max(initial_errors.max(), final_errors.max())],
                 'r--', alpha=0.7, label='No improvement line')

        ax6.set_xlabel('Initial Error')
        ax6.set_ylabel('Final Error')
        ax6.set_title(
            f'{title_prefix}: Per-Point Error Comparison\nImproved: {np.sum(final_errors < initial_errors)}/{len(final_errors)} points')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    filename = f"{title_prefix.replace(' ', '_').replace(':', '')}_transformation_analysis.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

    # Return summary statistics
    stats = {}
    if source_viz.shape[0] == transformed_viz.shape[0]:
        displacements = transformed_viz - source_viz
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        stats['displacement_mean'] = displacement_magnitudes.mean()
        stats['displacement_std'] = displacement_magnitudes.std()
        stats['displacement_max'] = displacement_magnitudes.max()

        if target_viz.shape[0] == transformed_viz.shape[0]:
            initial_errors = np.linalg.norm(source_viz - target_viz, axis=1)
            final_errors = np.linalg.norm(transformed_viz - target_viz, axis=1)
            stats['initial_rmse'] = np.sqrt(np.mean(initial_errors ** 2))
            stats['final_rmse'] = np.sqrt(np.mean(final_errors ** 2))
            stats['improved_points'] = np.sum(final_errors < initial_errors)
            stats['total_points'] = len(final_errors)

    return stats


def robust_svd(H, fallback_regularization=1e-8):
    """Robust SVD implementation with fallback methods for convergence issues."""
    try:
        U, S, Vt = np.linalg.svd(H)
        return U, S, Vt
    except np.linalg.LinAlgError as e:
        print(f"NumPy SVD failed: {e}. Trying scipy SVD with gesvd driver...")
        try:
            U, S, Vt = scipy_svd(H, lapack_driver='gesvd')
            return U, S, Vt
        except Exception as e2:
            print(f"Scipy SVD also failed: {e2}. Adding regularization...")
            try:
                H_reg = H + fallback_regularization * np.eye(min(H.shape))
                U, S, Vt = np.linalg.svd(H_reg)
                print(f"SVD succeeded with regularization {fallback_regularization}")
                return U, S, Vt
            except Exception as e3:
                print(f"All SVD methods failed: {e3}")
                raise


def rigid_alignment_procrustes(source, target):
    """Robust Procrustes alignment with improved error handling for SVD convergence issues."""
    # Input validation
    if source.shape[0] < 3 or target.shape[0] < 3:
        raise ValueError("Need at least 3 points for alignment")

    if source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("Points must be 3D")

    # Check for NaN or infinite values
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(target)):
        raise ValueError("Input contains NaN or infinite values")

    # Center both point clouds
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)

    # Check for degenerate configurations
    source_std = np.std(source_centered, axis=0)
    target_std = np.std(target_centered, axis=0)

    if np.any(source_std < 1e-10) or np.any(target_std < 1e-10):
        print("Warning: Degenerate point configuration detected")
        source_centered += np.random.normal(0, 1e-6, source_centered.shape)
        target_centered += np.random.normal(0, 1e-6, target_centered.shape)

    # If point clouds have different sizes, use closest point matching
    if source.shape[0] != target.shape[0]:
        from scipy.spatial.distance import cdist
        distances = cdist(source_centered, target_centered)
        closest_indices = np.argmin(distances, axis=1)
        target_matched = target_centered[closest_indices]
    else:
        target_matched = target_centered

    # Normalize for numerical stability
    source_scale = np.linalg.norm(source_centered, 'fro')
    target_scale = np.linalg.norm(target_matched, 'fro')

    if source_scale < 1e-10 or target_scale < 1e-10:
        print("Warning: Very small point cloud scale detected")
        return source.copy(), {
            'rotation_matrix': np.eye(3),
            'translation': np.zeros(3),
            'rmse_before': np.inf,
            'rmse_after': np.inf,
            'success': False
        }

    source_normalized = source_centered / source_scale
    target_normalized = target_matched / target_scale

    # Compute cross-covariance matrix
    H = source_normalized.T @ target_normalized

    # Check condition number
    cond_num = np.linalg.cond(H)
    if cond_num > 1e12:
        print(f"Warning: Ill-conditioned cross-covariance matrix (cond={cond_num:.2e})")

    # Robust SVD with multiple fallback methods
    try:
        U, S, Vt = robust_svd(H)
    except Exception as e:
        print(f"All SVD methods failed: {e}. Using identity transformation.")
        return source.copy(), {
            'rotation_matrix': np.eye(3),
            'translation': np.zeros(3),
            'rmse_before': np.inf,
            'rmse_after': np.inf,
            'success': False,
            'error': str(e)
        }

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply transformation with proper scaling
    aligned_source_centered = (R @ source_centered.T).T

    # Compute optimal translation
    t = np.mean(target, axis=0) - np.mean(aligned_source_centered + np.mean(source, axis=0), axis=0)

    # Final aligned source
    aligned_source = aligned_source_centered + np.mean(source, axis=0) + t

    # Calculate RMSE metrics
    rmse_before = np.sqrt(np.mean(
        np.sum((source - target_matched - np.mean(target, axis=0) + np.mean(target_matched, axis=0)) ** 2, axis=1)))
    rmse_after = np.sqrt(np.mean(
        np.sum((aligned_source - target_matched - np.mean(target, axis=0) + np.mean(target_matched, axis=0)) ** 2,
               axis=1)))

    transform_params = {
        'rotation_matrix': R,
        'translation': t,
        'rmse_before': rmse_before,
        'rmse_after': rmse_after,
        'success': True,
        'condition_number': cond_num
    }

    return aligned_source, transform_params


def compute_rmse(A, B):
    """Compute root mean square error between two point sets."""
    return np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1)))


def plot_point_sets(A, B, title="Point Cloud Comparison", A_label="Original", B_label="Target", save_dir=None):
    """Visualize two point clouds."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r', marker='o', label=A_label)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='b', marker='^', label=B_label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(title)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    else:
        plt.savefig(f"{title.replace(' ', '_')}.png")

    plt.close(fig)


def calculate_tps_transform(source_points, target_points):
    """Calculate a Thin Plate Spline transform between source and target points with error handling."""
    # Input validation
    if source_points.shape[0] != target_points.shape[0]:
        raise ValueError("Source and target must have same number of points")

    if np.any(~np.isfinite(source_points)) or np.any(~np.isfinite(target_points)):
        raise ValueError("Input contains NaN or infinite values")

    try:
        # Create separate RBF interpolators for each coordinate with smoothing
        rbf_x = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 0],
                    function='thin_plate', smooth=1e-6)
        rbf_y = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 1],
                    function='thin_plate', smooth=1e-6)
        rbf_z = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 2],
                    function='thin_plate', smooth=1e-6)

        def transform_function(points):
            try:
                x_transformed = rbf_x(points[:, 0], points[:, 1], points[:, 2])
                y_transformed = rbf_y(points[:, 0], points[:, 1], points[:, 2])
                z_transformed = rbf_z(points[:, 0], points[:, 1], points[:, 2])
                return np.vstack([x_transformed, y_transformed, z_transformed]).T
            except Exception as e:
                print(f"TPS transform evaluation failed: {e}")
                return points.copy()  # Return identity transform

        return transform_function

    except Exception as e:
        print(f"TPS transform creation failed: {e}")

        # Return identity transform as fallback
        def identity_transform(points):
            return points.copy()

        return identity_transform


def downsample_point_cloud(points, target_voxel_size=0.505):
    """Downsample a point cloud using Open3D's voxel downsampling method with error handling."""
    try:
        # Input validation
        if np.any(~np.isfinite(points)):
            print("Warning: Input contains NaN or infinite values")
            points = points[np.isfinite(points).all(axis=1)]

        if len(points) < 10:
            print("Warning: Very few input points for downsampling")
            return points

        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Perform voxel downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=target_voxel_size)

        # Convert back to numpy array
        downsampled_points = np.asarray(downsampled_pcd.points)

        print(f"Original points: {len(pcd.points)}, Downsampled points: {len(downsampled_pcd.points)}")

        # Ensure minimum number of points
        if len(downsampled_points) < 3:
            print("Error: Too few points after downsampling, using original points")
            return points

        if len(downsampled_points) < 10:
            print(f"Warning: Very few points after downsampling ({len(downsampled_points)})")

        return downsampled_points

    except Exception as e:
        print(f"Downsampling failed: {e}, using original points")
        return points


def create_results_visualization(specimen_name, deca_mean_source, aligned_test_target, deca_mean_transformed_pca,
                                 deca_mean_transformed_pca_v2,
                                 deca_mean_transformed_traditional, pca_rmse, pca_v2_rmse, traditional_rmse, save_dir):
    """Create a detailed visualization comparing all methods for a specimen."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Original points
    ax.scatter(deca_mean_source[:, 0], deca_mean_source[:, 1], deca_mean_source[:, 2],
               c='k', marker='o', s=30, label="Original DECA Mean")

    # Target points
    ax.scatter(aligned_test_target[:, 0], aligned_test_target[:, 1], aligned_test_target[:, 2],
               c='g', marker='*', s=50, label=f"{specimen_name} Target")

    # PCA transformed
    ax.scatter(deca_mean_transformed_pca[:, 0], deca_mean_transformed_pca[:, 1], deca_mean_transformed_pca[:, 2],
               c='r', marker='^', s=30, label=f"PCA-CPD (RMSE: {pca_rmse:.4f})")

    # PCA_V2 transformed
    ax.scatter(deca_mean_transformed_pca_v2[:, 0], deca_mean_transformed_pca_v2[:, 1],
               deca_mean_transformed_pca_v2[:, 2],
               c='m', marker='x', s=30, label=f"PCA-CPD-V2 (RMSE: {pca_v2_rmse:.4f})")

    # Traditional transformed
    ax.scatter(deca_mean_transformed_traditional[:, 0], deca_mean_transformed_traditional[:, 1],
               deca_mean_transformed_traditional[:, 2],
               c='b', marker='s', s=30, label=f"Traditional CPD (RMSE: {traditional_rmse:.4f})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Registration Methods Comparison - {specimen_name}")
    plt.savefig(os.path.join(save_dir, f"{specimen_name}_Comparison.png"))
    plt.close(fig)


def process_specimen(specimen_file, skull_source, mean_shape, U_reduced, eigenvalues, deca_mean_source, results_dir,
                     voxel_size=0.7, noise_params=None):
    """Process a single specimen file through all registration methods with downsampled target."""
    specimen_name = os.path.basename(specimen_file).split('.')[0]
    print(f"\n===== Processing {specimen_name} =====")

    # Create a specimen-specific output directory
    specimen_dir = os.path.join(results_dir, specimen_name)
    os.makedirs(specimen_dir, exist_ok=True)

    try:
        aligned_test_target = np.array([
            cp['position']
            for cp in json.load(open(specimen_file))["markups"][0]["controlPoints"]
        ])
    except Exception as e:
        print(f"Error loading {specimen_file}: {e}")
        return None

    # Load semilandmarks from .ply model instead of JSON
    try:
        ply_path = f"../data/aligned_models/{specimen_name}.ply_align.ply"
        pcd = o3d.io.read_point_cloud(ply_path)
        skull_target_full = np.asarray(pcd.points)
    except Exception as e:
        print(f"Error loading semilandmarks from PLY for {specimen_name}: {e}")
        return None

    # Downsample the target semilandmarks
    skull_target_downsampled = downsample_point_cloud(skull_target_full, voxel_size)

    print(
        f"Target resampling for {specimen_name}: {skull_target_full.shape[0]} points → {skull_target_downsampled.shape[0]} points (voxel size: {voxel_size})")

    # Add noise to downsampled target if noise parameters are provided
    if noise_params:
        print(f"Adding noise to downsampled target for {specimen_name}...")
        skull_target_noisy, transform_info = add_noise_to_point_cloud(
            skull_target_downsampled,
            **noise_params,
            seed=hash(specimen_name) % 2 ** 32  # Use specimen name as seed for reproducibility
        )

        print(f"Applied noise - Rotation: [{transform_info['rotation_angles_deg'][0]:.2f}, "
              f"{transform_info['rotation_angles_deg'][1]:.2f}, "
              f"{transform_info['rotation_angles_deg'][2]:.2f}]°, "
              f"Scale: {transform_info['scale_factor']:.3f}, "
              f"Translation: [{transform_info['translation'][0]:.2f}, "
              f"{transform_info['translation'][1]:.2f}, "
              f"{transform_info['translation'][2]:.2f}]")

        # Plot original vs noisy target
        plot_point_sets(skull_target_downsampled, skull_target_noisy,
                        title=f"Downsampled vs Noisy Target - {specimen_name}",
                        A_label="Downsampled Target",
                        B_label="Noisy Target",
                        save_dir=specimen_dir)

        # Rigid alignment to bring noisy target back to mean space
        print(f"Performing rigid alignment for {specimen_name}...")
        try:
            skull_target_aligned, alignment_params = rigid_alignment_procrustes(skull_target_noisy, skull_source)

            if alignment_params.get('success', False):
                print(
                    f"Rigid alignment RMSE improvement: {alignment_params['rmse_before']:.4f} → {alignment_params['rmse_after']:.4f}")

                # Plot noisy vs aligned target
                plot_point_sets(skull_target_noisy, skull_target_aligned,
                                title=f"Noisy vs Rigidly Aligned Target - {specimen_name}",
                                A_label="Noisy Target",
                                B_label="Rigidly Aligned Target",
                                save_dir=specimen_dir)

                # Use the aligned target for further processing
                skull_target = skull_target_aligned
            else:
                print("Rigid alignment failed, using noisy target")
                skull_target = skull_target_noisy

        except Exception as e:
            print(f"Rigid alignment failed with error: {e}, using noisy target")
            skull_target = skull_target_noisy
            alignment_params = {'success': False, 'error': str(e)}

    else:
        skull_target = skull_target_downsampled
        transform_info = None
        alignment_params = None

    # Plot original vs final processed target
    plot_point_sets(skull_target_full, skull_target,
                    title=f"Original vs Final Processed Target - {specimen_name}",
                    A_label="Original Target",
                    B_label="Final Processed Target",
                    save_dir=specimen_dir)

    # Calculate initial RMSE
    initial_rmse = compute_rmse(deca_mean_source, aligned_test_target)
    print(f"Initial RMSE between DECA mean and {specimen_name} target: {initial_rmse:.6f}")

    # Plot original data
    plot_point_sets(deca_mean_source, aligned_test_target,
                    title=f"Original: DECA Mean vs {specimen_name} Target",
                    A_label="DECA Mean (53 pts)",
                    B_label=f"{specimen_name} Target (53 pts)",
                    save_dir=specimen_dir)

    # Step 1: Run PCA-based CPD with processed target
    print(f"Running PCA-based CPD for {specimen_name} with processed target...")
    start_time = time.time()
    try:
        pca_reg = PCADeformableRegistration(
            X=skull_target,  # Processed Target (fixed)
            Y=skull_source,  # Source (moving)
            alpha=2,  # PCA parameter
            mean_shape=mean_shape,
            U=U_reduced,
            eigenvalues=eigenvalues,
            tolerance=0.001,
            w=0.1,  # EM parameter
            max_iterations=200
        )

        pca_transformed, _ = pca_reg.register()
        pca_time = time.time() - start_time
        print(f"PCA Time for {specimen_name}: {pca_time:.2f} seconds")

        # Visualize the semilandmark transformation
        print(f"Visualizing PCA semilandmark transformation for {specimen_name}...")
        semilandmark_stats = visualize_transformation(
            skull_source, skull_target, pca_transformed,
            f"PCA-CPD Semilandmarks - {specimen_name}",
            specimen_dir,
            source_label="Source Semilandmarks",
            target_label="Target Semilandmarks",
            transformed_label="PCA Transformed Semilandmarks",
            subsample=500  # Subsample for clearer visualization
        )
        print(f"PCA Semilandmark transformation stats: {semilandmark_stats}")

        # Calculate TPS transform for PCA result
        print(f"Calculating TPS transform from PCA registration for {specimen_name}...")
        pca_tps_transform = calculate_tps_transform(skull_source, pca_transformed)

        # Apply TPS transform to 53-point DECA mean
        deca_mean_transformed_pca = pca_tps_transform(deca_mean_source)

        # Visualize the DECA mean transformation
        print(f"Visualizing PCA DECA mean transformation for {specimen_name}...")
        deca_stats = visualize_transformation(
            deca_mean_source, aligned_test_target, deca_mean_transformed_pca,
            f"PCA-CPD DECA Mean - {specimen_name}",
            specimen_dir,
            source_label="Original DECA Mean",
            target_label="Target Landmarks",
            transformed_label="PCA Transformed DECA Mean"
        )
        print(f"PCA DECA transformation stats: {deca_stats}")

        # Calculate RMSE after PCA-based transformation
        pca_rmse = compute_rmse(deca_mean_transformed_pca, aligned_test_target)
        print(f"RMSE after PCA-based transformation (TPS) for {specimen_name}: {pca_rmse:.6f}")

        # Plot transformed vs target for PCA method
        plot_point_sets(deca_mean_transformed_pca, aligned_test_target,
                        title=f"PCA-CPD: Transformed DECA Mean vs {specimen_name}",
                        A_label="Transformed DECA Mean (PCA)",
                        B_label=f"{specimen_name} Target",
                        save_dir=specimen_dir)

        pca_success = True

    except Exception as e:
        print(f"PCA-CPD failed for {specimen_name}: {e}")
        pca_rmse = np.inf
        pca_time = 0
        deca_mean_transformed_pca = deca_mean_source.copy()
        pca_success = False

    # Step 2: Run PCA-V2 CPD with processed target
    print(f"Running PCA-V2 CPD for {specimen_name} with processed target...")
    start_time = time.time()
    try:
        pca_v2_reg = PCADeformableRegistration2(
            X=skull_target,  # Processed Target (fixed)
            Y=skull_source,  # Source (moving)
            alpha=0.1,  # PCA parameter
            mean_shape=mean_shape,
            U=U_reduced,
            eigenvalues=eigenvalues,
            tolerance=0.001,
            w=0.1,  # EM parameter
            max_iterations=150
        )

        pca_v2_transformed, _ = pca_v2_reg.register()
        pca_v2_time = time.time() - start_time
        print(f"PCA-V2 Time for {specimen_name}: {pca_v2_time:.2f} seconds")

        # Visualize the semilandmark transformation
        print(f"Visualizing PCA-V2 semilandmark transformation for {specimen_name}...")
        semilandmark_v2_stats = visualize_transformation(
            skull_source, skull_target, pca_v2_transformed,
            f"PCA-V2-CPD Semilandmarks - {specimen_name}",
            specimen_dir,
            source_label="Source Semilandmarks",
            target_label="Target Semilandmarks",
            transformed_label="PCA-V2 Transformed Semilandmarks",
            subsample=500
        )
        print(f"PCA-V2 Semilandmark transformation stats: {semilandmark_v2_stats}")

        # Calculate TPS transform for PCA-V2 result
        print(f"Calculating TPS transform from PCA-V2 registration for {specimen_name}...")
        pca_v2_tps_transform = calculate_tps_transform(skull_source, pca_v2_transformed)

        # Apply TPS transform to 53-point DECA mean
        deca_mean_transformed_pca_v2 = pca_v2_tps_transform(deca_mean_source)

        # Visualize the DECA mean transformation
        print(f"Visualizing PCA-V2 DECA mean transformation for {specimen_name}...")
        deca_v2_stats = visualize_transformation(
            deca_mean_source, aligned_test_target, deca_mean_transformed_pca_v2,
            f"PCA-V2-CPD DECA Mean - {specimen_name}",
            specimen_dir,
            source_label="Original DECA Mean",
            target_label="Target Landmarks",
            transformed_label="PCA-V2 Transformed DECA Mean"
        )
        print(f"PCA-V2 DECA transformation stats: {deca_v2_stats}")

        # Calculate RMSE after PCA-V2 CPD
        pca_v2_rmse = compute_rmse(deca_mean_transformed_pca_v2, aligned_test_target)
        print(f"RMSE after PCA-V2 CPD (direct transform) for {specimen_name}: {pca_v2_rmse:.6f}")

        # Plot transformed vs target for PCA-V2 method
        plot_point_sets(deca_mean_transformed_pca_v2, aligned_test_target,
                        title=f"PCA-V2-CPD: Transformed DECA Mean vs {specimen_name}",
                        A_label="Transformed DECA Mean (PCA-V2)",
                        B_label=f"{specimen_name} Target",
                        save_dir=specimen_dir)

        pca_v2_success = True

    except Exception as e:
        print(f"PCA-V2-CPD failed for {specimen_name}: {e}")
        pca_v2_rmse = np.inf
        pca_v2_time = 0
        deca_mean_transformed_pca_v2 = deca_mean_source.copy()
        pca_v2_success = False

    # Step 3: Run traditional CPD with processed target
    print(f"Running Traditional CPD for {specimen_name} with processed target...")
    start_time = time.time()
    try:
        traditional_reg = DeformableRegistration(
            X=skull_target,  # Processed Target (fixed)
            Y=skull_source,  # Source (moving)
            alpha=2,
            beta=1,  # gaussian kernel width
            tolerance=0.001,
            w=0.1,  # EM parameter
            max_iterations=200
        )

        traditional_transformed, _ = traditional_reg.register()
        traditional_time = time.time() - start_time
        print(f"Traditional Time for {specimen_name}: {traditional_time:.2f} seconds")

        # Visualize the semilandmark transformation
        print(f"Visualizing Traditional semilandmark transformation for {specimen_name}...")
        semilandmark_trad_stats = visualize_transformation(
            skull_source, skull_target, traditional_transformed,
            f"Traditional-CPD Semilandmarks - {specimen_name}",
            specimen_dir,
            source_label="Source Semilandmarks",
            target_label="Target Semilandmarks",
            transformed_label="Traditional Transformed Semilandmarks",
            subsample=500
        )
        print(f"Traditional Semilandmark transformation stats: {semilandmark_trad_stats}")

        # Calculate TPS transform for traditional result
        vanilla_tps_transform = calculate_tps_transform(skull_source, traditional_transformed)

        # Apply TPS transform to 53-point DECA mean
        deca_mean_transformed_traditional = vanilla_tps_transform(deca_mean_source)

        # Visualize the DECA mean transformation
        print(f"Visualizing Traditional DECA mean transformation for {specimen_name}...")
        deca_trad_stats = visualize_transformation(
            deca_mean_source, aligned_test_target, deca_mean_transformed_traditional,
            f"Traditional-CPD DECA Mean - {specimen_name}",
            specimen_dir,
            source_label="Original DECA Mean",
            target_label="Target Landmarks",
            transformed_label="Traditional Transformed DECA Mean"
        )
        print(f"Traditional DECA transformation stats: {deca_trad_stats}")

        # Calculate RMSE after traditional CPD
        traditional_rmse = compute_rmse(deca_mean_transformed_traditional, aligned_test_target)
        print(f"RMSE after traditional CPD (direct transform) for {specimen_name}: {traditional_rmse:.6f}")

        # Plot transformed vs target for traditional method
        plot_point_sets(deca_mean_transformed_traditional, aligned_test_target,
                        title=f"Traditional CPD: Transformed DECA Mean vs {specimen_name}",
                        A_label="Transformed DECA Mean (Traditional)",
                        B_label=f"{specimen_name} Target",
                        save_dir=specimen_dir)

        traditional_success = True

    except Exception as e:
        print(f"Traditional CPD failed for {specimen_name}: {e}")
        traditional_rmse = np.inf
        traditional_time = 0
        deca_mean_transformed_traditional = deca_mean_source.copy()
        traditional_success = False

    # Only create comparison plots if at least two methods succeeded
    successful_methods = sum([pca_success, pca_v2_success, traditional_success])
    if successful_methods >= 2:
        # Compare PCA vs Traditional methods
        if pca_success and traditional_success:
            plot_point_sets(deca_mean_transformed_pca, deca_mean_transformed_traditional,
                            title=f"PCA vs Traditional CPD - {specimen_name}",
                            A_label="PCA-CPD Transformed",
                            B_label="Traditional CPD Transformed",
                            save_dir=specimen_dir)

        # Compare PCA-V2 vs Traditional methods
        if pca_v2_success and traditional_success:
            plot_point_sets(deca_mean_transformed_pca_v2, deca_mean_transformed_traditional,
                            title=f"PCA-V2 vs Traditional CPD - {specimen_name}",
                            A_label="PCA-V2-CPD Transformed",
                            B_label="Traditional CPD Transformed",
                            save_dir=specimen_dir)

        # Compare PCA-V2 vs PCA
        if pca_success and pca_v2_success:
            plot_point_sets(deca_mean_transformed_pca, deca_mean_transformed_pca_v2,
                            title=f"PCA vs PCA-V2 CPD - {specimen_name}",
                            A_label="PCA-CPD Transformed",
                            B_label="PCA-V2 CPD Transformed",
                            save_dir=specimen_dir)

    # Calculate improvement metrics
    pca_improvement = ((initial_rmse - pca_rmse) / initial_rmse) * 100 if pca_rmse != np.inf else -np.inf
    pca_v2_improvement = ((initial_rmse - pca_v2_rmse) / initial_rmse) * 100 if pca_v2_rmse != np.inf else -np.inf
    traditional_improvement = ((
                                       initial_rmse - traditional_rmse) / initial_rmse) * 100 if traditional_rmse != np.inf else -np.inf

    # Determine best method
    valid_rmses = []
    method_names = []
    if pca_rmse != np.inf:
        valid_rmses.append(pca_rmse)
        method_names.append('PCA')
    if pca_v2_rmse != np.inf:
        valid_rmses.append(pca_v2_rmse)
        method_names.append('PCA-V2')
    if traditional_rmse != np.inf:
        valid_rmses.append(traditional_rmse)
        method_names.append('Traditional')

    if valid_rmses:
        best_idx = np.argmin(valid_rmses)
        best_method = method_names[best_idx]
    else:
        best_method = 'None (all failed)'

    # Calculate method differences
    if pca_rmse != np.inf and traditional_rmse != np.inf:
        pca_method_difference = abs(
            ((traditional_rmse - pca_rmse) / traditional_rmse) * 100 if traditional_rmse > pca_rmse else ((
                                                                                                                  pca_rmse - traditional_rmse) / pca_rmse) * 100)
        pca_better_method = 'PCA' if pca_rmse < traditional_rmse else 'Traditional'
    else:
        pca_method_difference = np.inf
        pca_better_method = 'Unknown'

    if pca_v2_rmse != np.inf and traditional_rmse != np.inf:
        pca_v2_method_difference = abs(
            ((traditional_rmse - pca_v2_rmse) / traditional_rmse) * 100 if traditional_rmse > pca_v2_rmse else ((
                                                                                                                        pca_v2_rmse - traditional_rmse) / pca_v2_rmse) * 100)
        pca_v2_better_method = 'PCA-V2' if pca_v2_rmse < traditional_rmse else 'Traditional'
    else:
        pca_v2_method_difference = np.inf
        pca_v2_better_method = 'Unknown'

    # Print summary for this specimen
    print(f"\n--- Summary for {specimen_name} ---")
    print(
        f"Target processing: {skull_target_full.shape[0]} points → {skull_target_downsampled.shape[0]} points (voxel size: {voxel_size})")

    if noise_params:
        print(f"Noise applied: Rotation ±{noise_params.get('rotation_std', 0):.1f}°, "
              f"Scale ±{noise_params.get('scale_std', 0):.2f}, "
              f"Translation ±{noise_params.get('translation_std', 0):.1f}")
        if alignment_params and alignment_params.get('success', False):
            print(f"Rigid alignment RMSE: {alignment_params['rmse_before']:.4f} → {alignment_params['rmse_after']:.4f}")

    print(f"Initial RMSE: {initial_rmse:.6f}")
    print(f"PCA-CPD RMSE: {pca_rmse:.6f} (Improvement: {pca_improvement:.2f}%) - Success: {pca_success}")
    print(f"PCA-CPD-V2 RMSE: {pca_v2_rmse:.6f} (Improvement: {pca_v2_improvement:.2f}%) - Success: {pca_v2_success}")
    print(
        f"Traditional CPD RMSE: {traditional_rmse:.6f} (Improvement: {traditional_improvement:.2f}%) - Success: {traditional_success}")

    if pca_method_difference != np.inf:
        print(
            f"Difference between PCA and traditional methods: {pca_method_difference:.2f}% ({pca_better_method} better)")
    if pca_v2_method_difference != np.inf:
        print(
            f"Difference between PCA-V2 and traditional methods: {pca_v2_method_difference:.2f}% ({pca_v2_better_method} better)")

    print(f"Best method: {best_method}")

    # Create the detailed comparison visualization if at least one method succeeded
    if successful_methods > 0:
        create_results_visualization(
            specimen_name,
            deca_mean_source,
            aligned_test_target,
            deca_mean_transformed_pca,
            deca_mean_transformed_pca_v2,
            deca_mean_transformed_traditional,
            pca_rmse,
            pca_v2_rmse,
            traditional_rmse,
            specimen_dir
        )

        # Create a bar chart for RMSE comparison
        plt.figure(figsize=(10, 6))
        methods = ['Initial', 'PCA-CPD', 'PCA-V2-CPD', 'Traditional CPD']
        rmse_values = [initial_rmse, pca_rmse if pca_rmse != np.inf else 0,
                       pca_v2_rmse if pca_v2_rmse != np.inf else 0,
                       traditional_rmse if traditional_rmse != np.inf else 0]
        colors = ['gray', 'red', 'purple', 'blue']

        # Create bar chart, but mark failed methods
        bars = plt.bar(methods, rmse_values, color=colors)

        # Add hatching for failed methods
        if pca_rmse == np.inf:
            bars[1].set_hatch('///')
        if pca_v2_rmse == np.inf:
            bars[2].set_hatch('///')
        if traditional_rmse == np.inf:
            bars[3].set_hatch('///')

        plt.ylabel('RMSE')
        title_suffix = "with Noise + Rigid Alignment" if noise_params else "Downsampled Target"
        plt.title(f'RMSE Comparison - {specimen_name} ({title_suffix})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add RMSE values on top of bars
        for i, v in enumerate(rmse_values):
            if v > 0:
                display_val = f'{v:.4f}' if v != np.inf else 'FAILED'
                plt.text(i, v + 0.01 if v != np.inf else 0.01, display_val, ha='center')

        plt.savefig(os.path.join(specimen_dir, f"{specimen_name}_RMSE_Comparison.png"))
        plt.close()

    # Return results for this specimen
    result = {
        'specimen': specimen_name,
        'original_points': skull_target_full.shape[0],
        'downsampled_points': skull_target_downsampled.shape[0],
        'downsampling_ratio': skull_target_downsampled.shape[0] / skull_target_full.shape[0],
        'initial_rmse': initial_rmse,
        'pca_rmse': pca_rmse if pca_rmse != np.inf else None,
        'pca_v2_rmse': pca_v2_rmse if pca_v2_rmse != np.inf else None,
        'traditional_rmse': traditional_rmse if traditional_rmse != np.inf else None,
        'pca_improvement': pca_improvement if pca_improvement != -np.inf else None,
        'pca_v2_improvement': pca_v2_improvement if pca_v2_improvement != -np.inf else None,
        'traditional_improvement': traditional_improvement if traditional_improvement != -np.inf else None,
        'pca_method_difference': pca_method_difference if pca_method_difference != np.inf else None,
        'pca_v2_method_difference': pca_v2_method_difference if pca_v2_method_difference != np.inf else None,
        'pca_better_method': pca_better_method,
        'pca_v2_better_method': pca_v2_better_method,
        'best_method': best_method,
        'pca_time': pca_time,
        'pca_v2_time': pca_v2_time,
        'traditional_time': traditional_time,
        'pca_success': pca_success,
        'pca_v2_success': pca_v2_success,
        'traditional_success': traditional_success
    }

    # Add noise-related information if applicable
    if noise_params:
        result.update({
            'noise_rotation_std': noise_params.get('rotation_std', 0),
            'noise_scale_std': noise_params.get('scale_std', 0),
            'noise_translation_std': noise_params.get('translation_std', 0),
            'rigid_alignment_rmse_before': alignment_params.get('rmse_before') if alignment_params else None,
            'rigid_alignment_rmse_after': alignment_params.get('rmse_after') if alignment_params else None,
            'rigid_alignment_success': alignment_params.get('success', False) if alignment_params else False
        })

    return result


def main():
    # Voxel size parameter for downsampling
    voxel_size = 0.505

    # Noise parameters - set to None to disable noise, or provide a dict to enable
    noise_params = None  # Disabled for transformation visualization

    # Uncomment the line below to enable noise
    # noise_params = {
    #     'rotation_std': 15.0,  # Standard deviation for rotation in degrees
    #     'scale_std': 0.1,  # Standard deviation for scaling factor
    #     'translation_std': 10.0  # Standard deviation for translation
    # }

    # Create results directory with appropriate naming
    if noise_params:
        results_dir = f"batch_comparison_robust_noisy_voxel_{voxel_size}_rot{noise_params['rotation_std']}_scale{noise_params['scale_std']}_trans{noise_params['translation_std']}"
    else:
        results_dir = f"batch_comparison_robust_downsampled_voxel_{voxel_size}_with_visualization"

    os.makedirs(results_dir, exist_ok=True)

    # Load the DECA mean source (same for all specimens)
    deca_mean_source = np.array([
        cp['position']
        for cp in json.load(open("../data/mean/decaMeanModel.mrk.json"))["markups"][0]["controlPoints"]
    ])

    # Load the skull source (mean semilandmarks)
    skull_source = np.array([
        cp["position"]
        for cp in json.load(open("../data/mean/semilandmarks.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    # Build the Statistical Shape Model (once for all specimens)
    print("Building Statistical Shape Model...")
    json_dir = "../data/semilandmarks/"
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    # Find all aligned landmark files
    aligned_dir = "../data/aligned_LMs/"
    aligned_files = [os.path.join(aligned_dir, f) for f in os.listdir(aligned_dir) if f.endswith(".mrk.json")]

    if not aligned_files:
        print(f"No .mrk.json files found in {aligned_dir}")
        return

    print(f"Found {len(aligned_files)} specimen files to process")

    if noise_params:
        print(f"Noise parameters: Rotation ±{noise_params['rotation_std']:.1f}°, "
              f"Scale ±{noise_params['scale_std']:.2f}, "
              f"Translation ±{noise_params['translation_std']:.1f}")
    else:
        print("No noise will be applied to targets")

    # Process each specimen with downsampled targets
    all_results = []
    failed_specimens = []

    for specimen_file in aligned_files[:10]:  # Process only first 2 specimens for visualization testing
        ###################################
        all_shapes = []
        for fname in json_files:
            substr = ".ply_"
            path_substr = "aligned_LMs/"
            # Get substring up to 'substr'
            json_specimen = fname.split(substr)[0]
            aligned_specimen = specimen_file.split(path_substr)[1]
            aligned_specimen = aligned_specimen.split(substr)[0]
            if json_specimen == aligned_specimen:
                continue
            path = os.path.join(json_dir, fname)
            with open(path, "r") as f:
                data = json.load(f)
            cpoints = data.get("markups", [])[0].get("controlPoints", [])
            arr = np.array([cp["position"] for cp in cpoints], dtype=float)
            all_shapes.append(arr)

        shapes_np = np.stack(all_shapes, axis=0)
        mean_shape, U_reduced, eigenvalues, num_modes = build_ssm(shapes_np, variance_threshold=0.95)
        print(f"Number of shape modes retained: {num_modes}")
        ###################################

        try:
            result = process_specimen(
                specimen_file,
                skull_source,
                mean_shape,
                U_reduced,
                eigenvalues,
                deca_mean_source,
                results_dir,
                voxel_size,
                noise_params
            )

            if result:
                all_results.append(result)
            else:
                failed_specimens.append(os.path.basename(specimen_file))

        except Exception as e:
            print(f"Fatal error processing {specimen_file}: {e}")
            failed_specimens.append(os.path.basename(specimen_file))

    # Print processing summary
    print(f"\n===== PROCESSING SUMMARY =====")
    print(f"Successfully processed: {len(all_results)} specimens")
    print(f"Failed to process: {len(failed_specimens)} specimens")
    if failed_specimens:
        print(f"Failed specimens: {', '.join(failed_specimens)}")

    print(f"\nResults and visualizations saved to {results_dir}")
    print("Check individual specimen folders for detailed transformation visualizations!")


if __name__ == "__main__":
    main()