import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycpd.pca_registration import PCADeformableRegistration
from pycpd.deformable_registration import DeformableRegistration
from pycpd.ssm import build_ssm
from scipy.interpolate import Rbf
import open3d as o3d


def compute_rmse(A, B):
    """Compute Root Mean Square Error between two point clouds."""
    return np.sqrt(np.mean((A - B) ** 2))


def plot_point_sets(A, B, title="Point Cloud Comparison", A_label="Original", B_label="Target"):
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
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def calculate_tps_transform(source_points, target_points):
    """Calculate a Thin Plate Spline transform between source and target points."""
    # Create separate RBF interpolators for each coordinate (x, y, z)
    rbf_x = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 0],
                function='thin_plate')
    rbf_y = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 1],
                function='thin_plate')
    rbf_z = Rbf(source_points[:, 0], source_points[:, 1], source_points[:, 2], target_points[:, 2],
                function='thin_plate')

    def transform_function(points):
        x_transformed = rbf_x(points[:, 0], points[:, 1], points[:, 2])
        y_transformed = rbf_y(points[:, 0], points[:, 1], points[:, 2])
        z_transformed = rbf_z(points[:, 0], points[:, 1], points[:, 2])
        return np.vstack([x_transformed, y_transformed, z_transformed]).T

    return transform_function


def repeat_preserving_original(points, num_target_points):
    """Keep original points intact and fill extra with nearest repeats."""
    num_original = points.shape[0]

    if num_original == num_target_points:
        return points

    repeated_points = np.tile(points, (num_target_points // num_original + 1, 1))[:num_target_points]
    return repeated_points


def downsample_point_cloud(points, target_voxel_size=0.05):
    """
    Downsample a point cloud using Open3D's voxel downsampling method
    with a fixed voxel size.

    Parameters:
    points (numpy.ndarray): The input point cloud array with shape (N, 3)
    target_voxel_size (float): The voxel size to use for downsampling

    Returns:
    numpy.ndarray: The downsampled point cloud
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform voxel downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=target_voxel_size)

    # Convert back to numpy array
    downsampled_points = np.asarray(downsampled_pcd.points)

    print(f"Original points: {len(pcd.points)}, Downsampled points: {len(downsampled_pcd.points)}")

    return downsampled_points

# open3d downsample


def main():
    # Load the data
    skull_source = np.array([
        cp["position"]
        for cp in json.load(open("../data/mean/semilandmarks.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    skull_target_full = np.array([
        cp["position"]
        for cp in json.load(open("../data/semilandmarks/A_J.ply_align.json"))["markups"][0]["controlPoints"]
    ], dtype=float)

    # Resample the target to have fewer points (e.g., 50% of original)

    skull_target = downsample_point_cloud(skull_target_full, 0.4)

    deca_mean_source = np.array([
        cp['position']
        for cp in json.load(open("../data/mean/decaMeanModel.mrk.json"))["markups"][0]["controlPoints"]
    ])

    aligned_test_target = np.array([
        cp['position']
        for cp in json.load(open("../data/aligned_LMs/A_J.ply_align.mrk.json"))["markups"][0]["controlPoints"]
    ])

    print(f"Skull source (mean semilandmarks) shape: {skull_source.shape}")
    print(f"Skull target FULL (A_J semilandmarks) shape: {skull_target_full.shape}")
    print(f"Skull target RESAMPLED (A_J semilandmarks) shape: {skull_target.shape}")
    print(f"DECA mean source (53 points) shape: {deca_mean_source.shape}")
    print(f"Aligned test target (53 points) shape: {aligned_test_target.shape}")

    # Plot original vs resampled target
    plot_point_sets(skull_target_full, skull_target,
                    title="Original vs Resampled Target",
                    A_label="Original Target",
                    B_label="Resampled Target")

    # Build the Statistical Shape Model
    json_dir = "../data/semilandmarks/"
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    all_shapes = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)
        cpoints = data.get("markups", [])[0].get("controlPoints", [])
        arr = np.array([cp["position"] for cp in cpoints], dtype=float)
        all_shapes.append(arr)

    shapes_np = np.stack(all_shapes, axis=0)
    print("Statistical Shape Model built from shapes:", shapes_np.shape)

    mean_shape, U_reduced, eigenvalues, num_modes = build_ssm(shapes_np, variance_threshold=0.95)
    print(f"Number of shape modes retained: {num_modes}")

    # Plot original data
    plot_point_sets(deca_mean_source, aligned_test_target,
                    title="Original Comparison: DECA Mean vs A_J Target (53 points)",
                    A_label="DECA Mean (53 pts)",
                    B_label="A_J Target (53 pts)")

    # Calculate initial RMSE
    initial_rmse = compute_rmse(deca_mean_source, aligned_test_target)
    print(f"Initial RMSE between DECA mean and A_J target (53 points): {initial_rmse:.6f}")

    # Step 1A: Run PCA-based transformation on the mean to target
    print("\n--- Running PCA-based CPD with Resampled Target ---")
    pca_reg = PCADeformableRegistration(
        X=skull_target,  # Resampled Target (fixed)
        Y=skull_source,  # Source (moving)
        alpha=0.1,  # PCA parameter
        mean_shape=mean_shape,
        U=U_reduced,
        eigenvalues=eigenvalues,
        tolerance=0.001,
        w=0.1,  # EM parameter
        max_iterations=150
    )

    pca_transformed, _ = pca_reg.register()

    # Step 2A: Calculate TPS transform based on original mean and transformed mean
    print("Calculating TPS transform from PCA registration results...")
    pca_tps_transform = calculate_tps_transform(skull_source, pca_transformed)

    # Step 3A: Apply the TPS transform to the 53-point DECA mean
    deca_mean_transformed_pca = pca_tps_transform(deca_mean_source)

    # Calculate RMSE after PCA-based transformation
    pca_rmse = compute_rmse(deca_mean_transformed_pca, aligned_test_target)
    print(f"RMSE after PCA-based transformation (TPS): {pca_rmse:.6f}")

    # Plot transformed vs target for PCA method
    plot_point_sets(deca_mean_transformed_pca, aligned_test_target,
                    title="PCA-CPD: Transformed DECA Mean vs A_J Target (Resampled)",
                    A_label="Transformed DECA Mean (PCA)",
                    B_label="A_J Target")

    # Step 4: Run traditional CPD
    print("\n--- Running Traditional CPD with Resampled Target ---")
    reg_vanilla = DeformableRegistration(
        X=skull_target,  # Resampled Target
        Y=skull_source,
    )

    traditional_transformed, _ = reg_vanilla.register()

    # Downsample the source to match target size for transformation with traditional method
    vanilla_tps_transform = calculate_tps_transform(skull_source, traditional_transformed)


    deca_mean_transformed_traditional = vanilla_tps_transform(deca_mean_source)



    print("Original 53 points shape:", deca_mean_source.shape)

    # Calculate RMSE after traditional transformation
    traditional_rmse = compute_rmse(deca_mean_transformed_traditional, aligned_test_target)
    print(f"RMSE after traditional CPD transformation (direct transform): {traditional_rmse:.6f}")

    # Plot transformed vs target for traditional method
    plot_point_sets(deca_mean_transformed_traditional, aligned_test_target,
                    title="Traditional CPD: Transformed DECA Mean vs A_J Target (Resampled)",
                    A_label="Transformed DECA Mean (Traditional)",
                    B_label="A_J Target")

    # Step 5: Compare PCA vs Traditional methods
    plot_point_sets(deca_mean_transformed_pca, deca_mean_transformed_traditional,
                    title="PCA vs Traditional CPD Comparison (Resampled Target)",
                    A_label="PCA-CPD Transformed",
                    B_label="Traditional CPD Transformed")

    # Compute comparison metrics
    pca_improvement = ((initial_rmse - pca_rmse) / initial_rmse) * 100
    traditional_improvement = ((initial_rmse - traditional_rmse) / initial_rmse) * 100
    method_difference = ((traditional_rmse - pca_rmse) / traditional_rmse) * 100 if traditional_rmse > pca_rmse else ((
                                                                                                                              pca_rmse - traditional_rmse) / pca_rmse) * 100

    # Summary
    print("\n--- Summary of Results (Resampled Target) ---")
    print(f"Target resampling: {skull_target_full.shape[0]} points → {skull_target.shape[0]} points")
    print(f"Initial RMSE: {initial_rmse:.6f}")
    print(f"PCA-CPD RMSE: {pca_rmse:.6f} (Improvement: {pca_improvement:.2f}%)")
    print(f"Traditional CPD RMSE: {traditional_rmse:.6f} (Improvement: {traditional_improvement:.2f}%)")
    print(
        f"Difference between methods: {abs(method_difference):.2f}% ({'PCA better' if pca_rmse < traditional_rmse else 'Traditional better'})")

    # Create comparison visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Original points
    ax.scatter(deca_mean_source[:, 0], deca_mean_source[:, 1], deca_mean_source[:, 2],
               c='k', marker='o', s=30, label="Original DECA Mean")

    # Target points
    ax.scatter(aligned_test_target[:, 0], aligned_test_target[:, 1], aligned_test_target[:, 2],
               c='g', marker='*', s=50, label="A_J Target")

    # PCA transformed
    ax.scatter(deca_mean_transformed_pca[:, 0], deca_mean_transformed_pca[:, 1], deca_mean_transformed_pca[:, 2],
               c='r', marker='^', s=30, label=f"PCA-CPD (RMSE: {pca_rmse:.4f})")

    # Traditional transformed
    ax.scatter(deca_mean_transformed_traditional[:, 0], deca_mean_transformed_traditional[:, 1],
               deca_mean_transformed_traditional[:, 2],
               c='b', marker='s', s=30, label=f"Traditional CPD (RMSE: {traditional_rmse:.4f})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Comparison of Registration Methods (Resampled Target)")
    plt.savefig("Registration_Methods_Comparison_Resampled.png")
    plt.show()

    # Create a bar chart for RMSE comparison
    plt.figure(figsize=(10, 6))
    methods = ['Initial', 'PCA-CPD', 'Traditional CPD']
    rmse_values = [initial_rmse, pca_rmse, traditional_rmse]
    colors = ['gray', 'red', 'blue']

    plt.bar(methods, rmse_values, color=colors)
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison Between Methods (Resampled Target)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add RMSE values on top of bars
    for i, v in enumerate(rmse_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.savefig("RMSE_Comparison_Resampled.png")
    plt.show()


if __name__ == "__main__":
    main()