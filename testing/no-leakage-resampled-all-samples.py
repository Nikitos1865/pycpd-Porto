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
import pandas as pd
from pathlib import Path
import open3d as o3d


def compute_rmse(A, B):
    """Compute Root Mean Square Error between two point clouds."""
    return np.sqrt(np.mean((A - B) ** 2))


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

    plt.close(fig)  # Close to avoid showing plots when running in batch mode


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


def downsample_point_cloud(points, target_voxel_size=0.4):
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
                     voxel_size=0.4):
    """Process a single specimen file through all registration methods with downsampled target."""
    specimen_name = os.path.basename(specimen_file).split('.')[0]
    print(f"\n===== Processing {specimen_name} =====")

    # Create a specimen-specific output directory
    specimen_dir = os.path.join(results_dir, specimen_name)
    os.makedirs(specimen_dir, exist_ok=True)

    # Load target landmarks (53 points - not downsampled)
    try:
        aligned_test_target = np.array([
            cp['position']
            for cp in json.load(open(specimen_file))["markups"][0]["controlPoints"]
        ])
    except Exception as e:
        print(f"Error loading {specimen_file}: {e}")
        return None

    # Load target semilandmarks
    try:
        skull_target_file = f"../data/semilandmarks/{specimen_name}.ply_align.json"
        skull_target_full = np.array([
            cp["position"]
            for cp in json.load(open(skull_target_file))["markups"][0]["controlPoints"]
        ], dtype=float)
    except Exception as e:
        print(f"Error loading semilandmarks for {specimen_name}: {e}")
        return None

    # Downsample the target semilandmarks
    skull_target = downsample_point_cloud(skull_target_full, voxel_size)

    print(
        f"Target resampling for {specimen_name}: {skull_target_full.shape[0]} points → {skull_target.shape[0]} points (voxel size: {voxel_size})")

    # Plot original vs downsampled target
    plot_point_sets(skull_target_full, skull_target,
                    title=f"Original vs Downsampled Target - {specimen_name}",
                    A_label="Original Target",
                    B_label="Downsampled Target",
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

    # Step 1: Run PCA-based CPD with downsampled target
    print(f"Running PCA-based CPD for {specimen_name} with downsampled target...")
    start_time = time.time()
    pca_reg = PCADeformableRegistration(
        X=skull_target,  # Downsampled Target (fixed)
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
    pca_time = time.time() - start_time

    print(f"PCA Time for {specimen_name}: {pca_time:.2f} seconds")

    # Step 2: Calculate TPS transform for PCA result
    print(f"Calculating TPS transform from PCA registration for {specimen_name}...")
    pca_tps_transform = calculate_tps_transform(skull_source, pca_transformed)

    # Step 3: Apply TPS transform to 53-point DECA mean
    deca_mean_transformed_pca = pca_tps_transform(deca_mean_source)

    # Calculate RMSE after PCA-based transformation
    pca_rmse = compute_rmse(deca_mean_transformed_pca, aligned_test_target)
    print(f"RMSE after PCA-based transformation (TPS) for {specimen_name}: {pca_rmse:.6f}")

    # Plot transformed vs target for PCA method
    plot_point_sets(deca_mean_transformed_pca, aligned_test_target,
                    title=f"PCA-CPD: Transformed DECA Mean vs {specimen_name}",
                    A_label="Transformed DECA Mean (PCA)",
                    B_label=f"{specimen_name} Target",
                    save_dir=specimen_dir)

    # Step 1: Run PCA-V2 CPD with downsampled target
    print(f"Running PCA-V2 CPD for {specimen_name} with downsampled target...")

    start_time = time.time()
    pca_v2_reg = PCADeformableRegistration2(
        X=skull_target,  # Downsampled Target (fixed)
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

    # Use transform_point_cloud for PCA-V2-CPD
    print(f"Calculating TPS transform from PCA-V2 registration for {specimen_name}...")

    pca_v2_tps_transform = calculate_tps_transform(skull_source, pca_v2_transformed)

    # Step 3: Apply TPS transform to 53-point DECA mean
    deca_mean_transformed_pca_v2 = pca_v2_tps_transform(deca_mean_source)

    # Calculate RMSE after PCA-V2 CPD
    pca_v2_rmse = compute_rmse(deca_mean_transformed_pca_v2, aligned_test_target)
    print(f"RMSE after PCA-V2 CPD (direct transform) for {specimen_name}: {pca_v2_rmse:.6f}")

    # Plot transformed vs target for PCA-V2 method
    plot_point_sets(deca_mean_transformed_pca_v2, aligned_test_target,
                    title=f"PCA-V2-CPD: Transformed DECA Mean vs {specimen_name}",
                    A_label="Transformed DECA Mean (PCA-V2)",
                    B_label=f"{specimen_name} Target",
                    save_dir=specimen_dir)

    # Step 4: Run traditional CPD with downsampled target
    print(f"Running Traditional CPD for {specimen_name} with downsampled target...")

    start_time = time.time()
    traditional_reg = DeformableRegistration(
        X=skull_target,  # Downsampled Target (fixed)
        Y=skull_source,  # Source (moving)
    )

    traditional_transformed, _ = traditional_reg.register()
    traditional_time = time.time() - start_time

    vanilla_tps_transform = calculate_tps_transform(skull_source, traditional_transformed)

    # Apply TPS transform to 53-point DECA mean
    deca_mean_transformed_traditional = vanilla_tps_transform(deca_mean_source)

    # Calculate RMSE after traditional CPD
    traditional_rmse = compute_rmse(deca_mean_transformed_traditional, aligned_test_target)
    print(f"RMSE after traditional CPD (direct transform) for {specimen_name}: {traditional_rmse:.6f}")

    # Plot transformed vs target for traditional method
    plot_point_sets(deca_mean_transformed_traditional, aligned_test_target,
                    title=f"Traditional CPD: Transformed DECA Mean vs {specimen_name}",
                    A_label="Transformed DECA Mean (Traditional)",
                    B_label=f"{specimen_name} Target",
                    save_dir=specimen_dir)

    # Compare PCA vs Traditional methods
    plot_point_sets(deca_mean_transformed_pca, deca_mean_transformed_traditional,
                    title=f"PCA vs Traditional CPD - {specimen_name}",
                    A_label="PCA-CPD Transformed",
                    B_label="Traditional CPD Transformed",
                    save_dir=specimen_dir)

    # Compare PCA-V2 vs Traditional methods
    plot_point_sets(deca_mean_transformed_pca_v2, deca_mean_transformed_traditional,
                    title=f"PCA-V2 vs Traditional CPD - {specimen_name}",
                    A_label="PCA-V2-CPD Transformed",
                    B_label="Traditional CPD Transformed",
                    save_dir=specimen_dir)

    # Compare PCA-V2 vs PCA
    plot_point_sets(deca_mean_transformed_pca, deca_mean_transformed_pca_v2,
                    title=f"PCA vs PCA-V2 CPD - {specimen_name}",
                    A_label="PCA-CPD Transformed",
                    B_label="PCA-V2 CPD Transformed",
                    save_dir=specimen_dir)

    # Calculate improvement metrics
    pca_improvement = ((initial_rmse - pca_rmse) / initial_rmse) * 100
    pca_v2_improvement = ((initial_rmse - pca_v2_rmse) / initial_rmse) * 100
    traditional_improvement = ((initial_rmse - traditional_rmse) / initial_rmse) * 100

    pca_method_difference = ((
                                         traditional_rmse - pca_rmse) / traditional_rmse) * 100 if traditional_rmse > pca_rmse else (
                                                                                                                                                (
                                                                                                                                                            pca_rmse - traditional_rmse) / pca_rmse) * 100
    pca_better_method = 'PCA' if pca_rmse < traditional_rmse else 'Traditional'

    pca_v2_method_difference = ((
                                            traditional_rmse - pca_v2_rmse) / traditional_rmse) * 100 if traditional_rmse > pca_v2_rmse else (
                                                                                                                                                         (
                                                                                                                                                                     pca_v2_rmse - traditional_rmse) / pca_v2_rmse) * 100
    pca_v2_better_method = 'PCA-V2' if pca_v2_rmse < traditional_rmse else 'Traditional'

    if min(pca_rmse, pca_v2_rmse, traditional_rmse) == pca_rmse:
        best_method = 'PCA'
    elif min(pca_rmse, pca_v2_rmse, traditional_rmse) == pca_v2_rmse:
        best_method = 'PCA-V2'
    else:
        best_method = 'Traditional'

    # Print summary for this specimen
    print(f"\n--- Summary for {specimen_name} ---")
    print(
        f"Target downsampling: {skull_target_full.shape[0]} points → {skull_target.shape[0]} points (voxel size: {voxel_size})")
    print(f"Initial RMSE: {initial_rmse:.6f}")
    print(f"PCA-CPD RMSE: {pca_rmse:.6f} (Improvement: {pca_improvement:.2f}%)")
    print(f"PCA-CPD-V2 RMSE: {pca_v2_rmse:.6f} (Improvement: {pca_v2_improvement:.2f}%)")
    print(f"Traditional CPD RMSE: {traditional_rmse:.6f} (Improvement: {traditional_improvement:.2f}%)")
    print(
        f"Difference between PCA and traditional methods: {abs(pca_method_difference):.2f}% ({pca_better_method} better)")
    print(
        f"Difference between PCA-V2 and traditional methods: {abs(pca_v2_method_difference):.2f}% ({pca_v2_better_method} better)")
    print(f"Best method: {best_method}")

    # Create the detailed comparison visualization
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
    rmse_values = [initial_rmse, pca_rmse, pca_v2_rmse, traditional_rmse]
    colors = ['gray', 'red', 'purple', 'blue']

    plt.bar(methods, rmse_values, color=colors)
    plt.ylabel('RMSE')
    plt.title(f'RMSE Comparison - {specimen_name} (Downsampled Target)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add RMSE values on top of bars
    for i, v in enumerate(rmse_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.savefig(os.path.join(specimen_dir, f"{specimen_name}_RMSE_Comparison.png"))
    plt.close()

    # Return results for this specimen
    return {
        'specimen': specimen_name,
        'original_points': skull_target_full.shape[0],
        'downsampled_points': skull_target.shape[0],
        'downsampling_ratio': skull_target.shape[0] / skull_target_full.shape[0],
        'initial_rmse': initial_rmse,
        'pca_rmse': pca_rmse,
        'pca_v2_rmse': pca_v2_rmse,
        'traditional_rmse': traditional_rmse,
        'pca_improvement': pca_improvement,
        'pca_v2_improvement': pca_v2_improvement,
        'traditional_improvement': traditional_improvement,
        'pca_method_difference': abs(pca_method_difference),
        'pca_v2_method_difference': abs(pca_v2_method_difference),
        'pca_better_method': pca_better_method,
        'pca_v2_better_method': pca_v2_better_method,
        'best_method': best_method,
        'pca_time': pca_time,
        'pca_v2_time': pca_v2_time,
        'traditional_time': traditional_time
    }


def main():
    # Voxel size parameter for downsampling
    voxel_size = 0.4  # Same as in paste-2.txt

    # Create results directory with voxel size in name
    results_dir = f"batch_comparison_downsampled_voxel_{voxel_size}"
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

    # Process each specimen with downsampled targets
    all_results = []
    for specimen_file in aligned_files:
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
        result = process_specimen(
            specimen_file,
            skull_source,
            mean_shape,
            U_reduced,
            eigenvalues,
            deca_mean_source,
            results_dir,
            voxel_size
        )

        if result:
            all_results.append(result)

    # Create a summary dataframe
    if all_results:
        df = pd.DataFrame(all_results)

        # Save to CSV
        df.to_csv(os.path.join(results_dir, f"batch_comparison_downsampled_voxel_{voxel_size}.csv"), index=False)

        # Print summary statistics with format similar to paste-2.txt example
        print("\n===== OVERALL RESULTS WITH DOWNSAMPLED TARGETS =====")
        print(f"Voxel size for downsampling: {voxel_size}")
        print(f"Total specimens processed: {len(all_results)}")

        # Summary of downsampling
        avg_original = df['original_points'].mean()
        avg_downsampled = df['downsampled_points'].mean()
        avg_ratio = df['downsampling_ratio'].mean()
        print(
            f"Target resampling: Average {avg_original:.1f} points → {avg_downsampled:.1f} points (ratio: {avg_ratio:.2f})")

        # RMSE metrics
        print(f"Average Initial RMSE: {df['initial_rmse'].mean():.6f}")
        print(f"Average PCA-CPD RMSE: {df['pca_rmse'].mean():.6f}")
        print(f"Average PCA-V2-CPD RMSE: {df['pca_v2_rmse'].mean():.6f}")
        print(f"Average Traditional CPD RMSE: {df['traditional_rmse'].mean():.6f}")

        # Improvement metrics
        print(f"Average PCA Improvement: {df['pca_improvement'].mean():.2f}%")
        print(f"Average PCA-V2 Improvement: {df['pca_v2_improvement'].mean():.2f}%")
        print(f"Average Traditional Improvement: {df['traditional_improvement'].mean():.2f}%")

        # Method comparison
        print(f"Average difference between PCA and Traditional: {df['pca_method_difference'].mean():.2f}%")
        print(f"Average difference between PCA-V2 and Traditional: {df['pca_v2_method_difference'].mean():.2f}%")

        # Timing metrics
        print(f"Average PCA-CPD Time: {df['pca_time'].mean():.4f} seconds")
        print(f"Average PCA-V2 Time: {df['pca_v2_time'].mean():.4f} seconds")
        print(f"Average Traditional CPD Time: {df['traditional_time'].mean():.4f} seconds")

        # Count which method performed better
        method_counts = df['best_method'].value_counts()
        for method, count in method_counts.items():
            print(f"{method} was better in {count} cases ({count / len(df) * 100:.2f}%)")

        # Create overall comparison chart
        plt.figure(figsize=(12, 8))

        # Sort by initial RMSE for better visualization
        df_sorted = df.sort_values('initial_rmse')

        x = range(len(df_sorted))
        width = 0.2  # Adjusted for 4 bars

        plt.bar([i - 1.5 * width for i in x], df_sorted['initial_rmse'], width, label='Initial RMSE', color='gray')
        plt.bar([i - 0.5 * width for i in x], df_sorted['pca_rmse'], width, label='PCA-CPD RMSE', color='red')
        plt.bar([i + 0.5 * width for i in x], df_sorted['pca_v2_rmse'], width, label='PCA-V2-CPD RMSE', color='purple')
        plt.bar([i + 1.5 * width for i in x], df_sorted['traditional_rmse'], width, label='Traditional CPD RMSE',
                color='blue')

        plt.xlabel('Specimen')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Comparison Across All Specimens (Downsampled, Voxel Size {voxel_size})')
        plt.xticks([i for i in x], df_sorted['specimen'], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "Overall_RMSE_Comparison.png"))
        plt.close()

        # Create improvement comparison
        plt.figure(figsize=(12, 8))

        # Sort by PCA improvement for better visualization
        df_sorted = df.sort_values('pca_improvement')

        x = range(len(df_sorted))
        width = 0.25

        plt.bar([i - width for i in x], df_sorted['pca_improvement'], width, label='PCA-CPD Improvement', color='red')
        plt.bar([i for i in x], df_sorted['pca_v2_improvement'], width, label='PCA-V2-CPD Improvement', color='purple')
        plt.bar([i + width for i in x], df_sorted['traditional_improvement'], width,
                label='Traditional CPD Improvement', color='blue')

        plt.xlabel('Specimen')
        plt.ylabel('Improvement (%)')
        plt.title(f'Registration Improvement Across All Specimens (Downsampled, Voxel Size {voxel_size})')
        plt.xticks([i for i in x], df_sorted['specimen'], rotation=90)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "Overall_Improvement_Comparison.png"))
        plt.close()

        # Create time comparison bar chart
        plt.figure(figsize=(10, 6))
        methods = ['PCA-CPD', 'PCA-V2-CPD', 'Traditional CPD']
        avg_times = [df['pca_time'].mean(), df['pca_v2_time'].mean(), df['traditional_time'].mean()]
        colors = ['red', 'purple', 'blue']

        plt.bar(methods, avg_times, color=colors)
        plt.ylabel('Average Time (seconds)')
        plt.title(f'Average Processing Time Comparison (Downsampled, Voxel Size {voxel_size})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add time values on top of bars
        for i, v in enumerate(avg_times):
            plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')

        plt.savefig(os.path.join(results_dir, "Average_Time_Comparison.png"))
        plt.close()

        # Create a scatterplot of downsampling ratio vs improvement
        plt.figure(figsize=(12, 8))
        plt.scatter(df['downsampling_ratio'], df['pca_improvement'], color='red', label='PCA-CPD Improvement')
        plt.scatter(df['downsampling_ratio'], df['pca_v2_improvement'], color='purple', label='PCA-V2-CPD Improvement')
        plt.scatter(df['downsampling_ratio'], df['traditional_improvement'], color='blue',
                    label='Traditional CPD Improvement')

        plt.xlabel('Downsampling Ratio (downsampled/original)')
        plt.ylabel('Improvement (%)')
        plt.title(f'Effect of Downsampling Ratio on Registration Improvement (Voxel Size {voxel_size})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "Downsampling_vs_Improvement.png"))
        plt.close()

        # Create a stacked bar chart showing best method counts
        plt.figure(figsize=(8, 6))
        method_labels = method_counts.index.tolist()
        method_values = method_counts.values.tolist()
        method_colors = {'PCA': 'red', 'PCA-V2': 'purple', 'Traditional': 'blue'}
        colors = [method_colors[method] for method in method_labels]

        plt.bar(method_labels, method_values, color=colors)
        plt.ylabel('Number of Specimens')
        plt.title('Best Performance by Method')

        # Add count and percentage on top of bars
        for i, v in enumerate(method_values):
            plt.text(i, v + 0.5, f'{v} ({v / len(df) * 100:.1f}%)', ha='center')

        plt.savefig(os.path.join(results_dir, "Best_Method_Comparison.png"))
        plt.close()

        # Print comparative summary based on voxel size
        print("\n--- DOWNSAMPLING EFFECT SUMMARY ---")
        print(f"With voxel size {voxel_size}, average points reduced by {(1 - avg_ratio) * 100:.1f}%")

        # Determine if downsampling improved performance
        improved_count = sum(df['best_method'] != 'Traditional')

        print(
            f"PCA-based methods (PCA, PCA-V2) performed better than Traditional in {improved_count} out of {len(df)} cases ({improved_count / len(df) * 100:.1f}%)")
        print(
            f"Average processing time: PCA-CPD = {df['pca_time'].mean():.2f}s, PCA-V2 = {df['pca_v2_time'].mean():.2f}s, Traditional = {df['traditional_time'].mean():.2f}s")

        if df['pca_time'].mean() < df['traditional_time'].mean():
            time_savings = (df['traditional_time'].mean() - df['pca_time'].mean()) / df['traditional_time'].mean() * 100
            print(f"PCA-CPD was {time_savings:.1f}% faster than Traditional CPD")

        if df['pca_v2_time'].mean() < df['traditional_time'].mean():
            time_savings = (df['traditional_time'].mean() - df['pca_v2_time'].mean()) / df[
                'traditional_time'].mean() * 100
            print(f"PCA-V2-CPD was {time_savings:.1f}% faster than Traditional CPD")

        print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()