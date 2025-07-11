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
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

from pycpd.utility import compute_rmse, downsample_point_cloud, calculate_tps_transform


def detailed_visualization_step(points_dict, title, save_path, step_num):
    """Create detailed visualization for a specific step."""

    # Get all point sets
    point_sets = list(points_dict.keys())
    n_sets = len(point_sets)

    # Determine grid layout based on number of datasets
    if n_sets <= 2:
        rows, cols = 1, 2
    elif n_sets <= 4:
        rows, cols = 2, 2
    elif n_sets <= 6:
        rows, cols = 2, 3
    elif n_sets <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4  # Max 12 subplots

    fig = plt.figure(figsize=(cols * 6, rows * 5))

    # Use different colors and markers for different point sets
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', '^', 's', 'D', 'v', 'p', '*', 'h', '+', 'x']

    # Create individual plots for each dataset (limit to available subplot slots)
    max_individual_plots = min(n_sets, rows * cols - 1)  # Reserve one for overlay

    for i, (name, points) in enumerate(list(points_dict.items())[:max_individual_plots]):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Subsample for visualization if too many points
        if len(points) > 1000:
            indices = np.random.choice(len(points), 1000, replace=False)
            viz_points = points[indices]
            ax.set_title(f"{name}\n({len(points)} points, showing 1000)", fontsize=8)
        else:
            viz_points = points
            ax.set_title(f"{name}\n({len(points)} points)", fontsize=8)

        ax.scatter(viz_points[:, 0], viz_points[:, 1], viz_points[:, 2],
                   c=color, marker=marker, s=20, alpha=0.7)

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=6)

    # Create overlay plot if we have multiple datasets and space for it
    if n_sets >= 2 and max_individual_plots < rows * cols:
        ax_overlay = fig.add_subplot(rows, cols, max_individual_plots + 1, projection='3d')

        for i, (name, points) in enumerate(points_dict.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # Subsample for clarity
            if len(points) > 500:
                indices = np.random.choice(len(points), 500, replace=False)
                viz_points = points[indices]
            else:
                viz_points = points

            ax_overlay.scatter(viz_points[:, 0], viz_points[:, 1], viz_points[:, 2],
                               c=color, marker=marker, s=15, alpha=0.6, label=name[:15])  # Truncate long names

        ax_overlay.set_title("Overlay Comparison", fontsize=8)
        ax_overlay.legend(fontsize=6, loc='upper right')
        ax_overlay.set_xlabel('X', fontsize=8)
        ax_overlay.set_ylabel('Y', fontsize=8)
        ax_overlay.set_zlabel('Z', fontsize=8)
        ax_overlay.tick_params(labelsize=6)

    plt.suptitle(f"Step {step_num}: {title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {save_path}")


def analyze_transformation_quality(source, transformed, target, name):
    """Analyze the quality of a transformation."""
    print(f"\n=== {name} Transformation Analysis ===")

    # Calculate displacements
    displacements = transformed - source
    displacement_mags = np.linalg.norm(displacements, axis=1)

    print(f"Displacement Statistics:")
    print(f"  Mean: {displacement_mags.mean():.6f}")
    print(f"  Std:  {displacement_mags.std():.6f}")
    print(f"  Max:  {displacement_mags.max():.6f}")
    print(f"  Min:  {displacement_mags.min():.6f}")

    # Calculate errors if target is available and same size
    if target is not None and len(target) == len(transformed):
        initial_errors = np.linalg.norm(source - target, axis=1)
        final_errors = np.linalg.norm(transformed - target, axis=1)

        initial_rmse = np.sqrt(np.mean(initial_errors ** 2))
        final_rmse = np.sqrt(np.mean(final_errors ** 2))
        improvement = ((initial_rmse - final_rmse) / initial_rmse) * 100

        improved_points = np.sum(final_errors < initial_errors)

        print(f"Accuracy Statistics:")
        print(f"  Initial RMSE: {initial_rmse:.6f}")
        print(f"  Final RMSE:   {final_rmse:.6f}")
        print(f"  Improvement:  {improvement:.2f}%")
        print(
            f"  Improved points: {improved_points}/{len(final_errors)} ({improved_points / len(final_errors) * 100:.1f}%)")

        return {
            'displacement_mean': displacement_mags.mean(),
            'displacement_std': displacement_mags.std(),
            'displacement_max': displacement_mags.max(),
            'initial_rmse': initial_rmse,
            'final_rmse': final_rmse,
            'improvement_pct': improvement,
            'improved_points': improved_points,
            'total_points': len(final_errors)
        }
    else:
        return {
            'displacement_mean': displacement_mags.mean(),
            'displacement_std': displacement_mags.std(),
            'displacement_max': displacement_mags.max()
        }


def analyze_negative_samples():
    """Analyze the specific negative-performing samples identified."""

    # Define the problematic samples from your data
    negative_samples = [
        "129X1_SVJ",  # pca_improvement: -31.82, traditional_improvement: -3.94
        "AKR_J",  # pca_improvement: -12.10, pca_v2_improvement: -12.44, traditional_improvement: -28.80
        "A_J",  # pca_improvement: -4.18, traditional_improvement: -10.29
        "BALB_CBYJ",  # pca_improvement: -7.86, traditional_improvement: -18.51
        "BUB",  # pca_improvement: -5.00, traditional_improvement: -12.42
        "CBA_CAJ",  # pca_improvement: -47.54, pca_v2_improvement: -7.996, traditional_improvement: -29.44
        "CBA_J",  # pca_improvement: -6.77, traditional_improvement: -11.61
        "I",  # pca_improvement: -7.39, pca_v2_improvement: -6.42
        "PL",  # pca_improvement: -10.94, traditional_improvement: -4.40
        "SJL_J",  # pca_improvement: -19.39, pca_v2_improvement: -16.91
        "SWR",  # pca_improvement: -5.59
        "SM"  # traditional_improvement: -12.10
    ]

    # Priority order: worst cases first
    priority_order = ["CBA_CAJ", "129X1_SVJ", "AKR_J", "SJL_J", "PL", "BALB_CBYJ"]

    # Create output directory
    output_dir = "negative_samples_analysis"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("DETAILED ANALYSIS OF NEGATIVE-PERFORMING SAMPLES")
    print("=" * 80)

    # Store all results for comparison
    all_results = {}

    # Load DECA mean landmarks (53 points) - this is constant
    deca_mean_source = np.array([
        cp['position']
        for cp in json.load(open("../data/mean/decaMeanModel.mrk.json"))["markups"][0]["controlPoints"]
    ])
    print(f"DECA mean landmarks loaded: {deca_mean_source.shape}")

    # Load mean semilandmarks - this is constant
    skull_source = np.array([
        cp["position"]
        for cp in json.load(open("../data/mean/semilandmarks.json"))["markups"][0]["controlPoints"]
    ], dtype=float)
    print(f"Mean semilandmarks loaded: {skull_source.shape}")

    # Analyze each problematic sample
    for i, specimen_name in enumerate(priority_order):
        if specimen_name not in negative_samples:
            continue

        print(f"\n\n{'=' * 80}")
        print(f"ANALYZING SPECIMEN {i + 1}/{len(priority_order)}: {specimen_name}")
        print(f"{'=' * 80}")

        try:
            # Create specimen-specific output directory
            specimen_dir = os.path.join(output_dir, specimen_name)
            os.makedirs(specimen_dir, exist_ok=True)

            # Find the specimen file
            aligned_dir = "../data/aligned_LMs/"
            specimen_file = None
            for f in os.listdir(aligned_dir):
                if f.startswith(specimen_name) and f.endswith(".mrk.json"):
                    specimen_file = os.path.join(aligned_dir, f)
                    break

            if specimen_file is None:
                print(f"❌ Could not find specimen file for {specimen_name}")
                continue

            print(f"📁 Found specimen file: {os.path.basename(specimen_file)}")

            # Load target landmarks (53 points)
            target_landmarks = np.array([
                cp['position']
                for cp in json.load(open(specimen_file))["markups"][0]["controlPoints"]
            ])
            print(f"Target landmarks loaded: {target_landmarks.shape}")

            # Load target semilandmarks from PLY
            ply_path = f"../data/aligned_models/{specimen_name}.ply_align.ply"
            if not os.path.exists(ply_path):
                print(f"❌ Could not find PLY file: {ply_path}")
                continue

            pcd = o3d.io.read_point_cloud(ply_path)
            target_semilandmarks_full = np.asarray(pcd.points)
            print(f"Target semilandmarks loaded: {target_semilandmarks_full.shape}")

            # Calculate initial baseline
            initial_landmark_rmse = compute_rmse(deca_mean_source, target_landmarks)
            print(f"🎯 Initial DECA mean vs target landmarks RMSE: {initial_landmark_rmse:.6f}")

            # Downsample target semilandmarks
            target_semilandmarks_downsampled = downsample_point_cloud(target_semilandmarks_full, 0.505)
            downsampling_ratio = len(target_semilandmarks_downsampled) / len(target_semilandmarks_full)
            print(f"Downsampling ratio: {downsampling_ratio:.3f}")

            # Build SSM using leave-one-out methodology
            print(f"\n🔬 Building SSM (excluding {specimen_name})...")
            json_dir = "../data/semilandmarks/"
            json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

            all_shapes = []
            excluded_specimen = None

            for fname in json_files:
                substr = ".ply_"
                json_specimen = fname.split(substr)[0]

                # Check if this matches our target specimen
                if json_specimen == specimen_name:
                    excluded_specimen = fname
                    print(f"  Excluding {json_specimen} from SSM (leave-one-out)")
                    continue

                # Load the semilandmark data
                path = os.path.join(json_dir, fname)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    cpoints = data.get("markups", [])[0].get("controlPoints", [])
                    arr = np.array([cp["position"] for cp in cpoints], dtype=float)
                    all_shapes.append(arr)
                except Exception as e:
                    print(f"  Warning: Could not load {fname}: {e}")
                    continue

            if not all_shapes:
                print(f"❌ No valid shapes loaded for SSM construction!")
                continue

            print(f"  SSM construction: Using {len(all_shapes)} specimens")

            shapes_np = np.stack(all_shapes, axis=0)
            mean_shape, U_reduced, eigenvalues, num_modes = build_ssm(shapes_np, variance_threshold=0.95)
            print(f"  SSM built: {num_modes} modes retained")

            # Analyze data characteristics for this specimen
            print(f"\n📊 DATA CHARACTERISTICS FOR {specimen_name}:")

            # Point cloud density analysis
            bounds_target = np.ptp(target_semilandmarks_downsampled, axis=0)
            bounds_mean = np.ptp(skull_source, axis=0)

            print(f"  Target semilandmarks bounds: {bounds_target}")
            print(f"  Mean semilandmarks bounds: {bounds_mean}")
            print(f"  Scale difference: {bounds_target / bounds_mean}")

            # Distance distribution analysis
            target_centroid = np.mean(target_semilandmarks_downsampled, axis=0)
            mean_centroid = np.mean(skull_source, axis=0)
            centroid_distance = np.linalg.norm(target_centroid - mean_centroid)
            print(f"  Centroid distance: {centroid_distance:.6f}")

            # Shape complexity (variance in distances from centroid)
            target_distances = np.linalg.norm(target_semilandmarks_downsampled - target_centroid, axis=1)
            mean_distances = np.linalg.norm(skull_source - mean_centroid, axis=1)

            print(f"  Target shape complexity (std of radial distances): {np.std(target_distances):.6f}")
            print(f"  Mean shape complexity (std of radial distances): {np.std(mean_distances):.6f}")

            # Now run registration methods
            print(f"\n🚀 RUNNING REGISTRATION METHODS...")
            results = {}

            # PCA-CPD (v1)
            print("\n--- PCA-CPD (v1) ---")
            start_time = time.time()
            try:
                pca_reg = PCADeformableRegistration(
                    X=target_semilandmarks_downsampled,  # Target (fixed)
                    Y=skull_source,  # Source (moving)
                    alpha=2,
                    mean_shape=mean_shape,
                    U=U_reduced,
                    eigenvalues=eigenvalues,
                    tolerance=0.001,
                    w=0.1,
                    max_iterations=200
                )

                pca_transformed, _ = pca_reg.register()
                pca_time = time.time() - start_time
                print(f"  ✅ PCA-CPD completed in {pca_time:.2f} seconds")

                # Analyze PCA transformation
                pca_semi_stats = analyze_transformation_quality(
                    skull_source, pca_transformed, target_semilandmarks_downsampled, "PCA Semilandmark"
                )

                # Create TPS and apply to landmarks
                pca_tps_transform = calculate_tps_transform(skull_source, pca_transformed)
                pca_deca_transformed = pca_tps_transform(deca_mean_source)

                pca_landmark_stats = analyze_transformation_quality(
                    deca_mean_source, pca_deca_transformed, target_landmarks, "PCA Landmark"
                )

                results['pca'] = {
                    'semilandmarks_transformed': pca_transformed,
                    'landmarks_transformed': pca_deca_transformed,
                    'time': pca_time,
                    'semi_stats': pca_semi_stats,
                    'landmark_stats': pca_landmark_stats,
                    'success': True
                }

            except Exception as e:
                print(f"  ❌ PCA-CPD failed: {e}")
                results['pca'] = {'success': False, 'error': str(e)}

            # PCA-CPD-V2
            print("\n--- PCA-CPD-V2 ---")
            start_time = time.time()
            try:
                pca_v2_reg = PCADeformableRegistration2(
                    X=target_semilandmarks_downsampled,  # Target (fixed)
                    Y=skull_source,  # Source (moving)
                    alpha=0.1,
                    mean_shape=mean_shape,
                    U=U_reduced,
                    eigenvalues=eigenvalues,
                    tolerance=0.001,
                    w=0.1,
                    max_iterations=150
                )

                pca_v2_transformed, _ = pca_v2_reg.register()
                pca_v2_time = time.time() - start_time
                print(f"  ✅ PCA-CPD-V2 completed in {pca_v2_time:.2f} seconds")

                # Analyze PCA-V2 transformation
                pca_v2_semi_stats = analyze_transformation_quality(
                    skull_source, pca_v2_transformed, target_semilandmarks_downsampled, "PCA-V2 Semilandmark"
                )

                # Create TPS and apply to landmarks
                pca_v2_tps_transform = calculate_tps_transform(skull_source, pca_v2_transformed)
                pca_v2_deca_transformed = pca_v2_tps_transform(deca_mean_source)

                pca_v2_landmark_stats = analyze_transformation_quality(
                    deca_mean_source, pca_v2_deca_transformed, target_landmarks, "PCA-V2 Landmark"
                )

                results['pca_v2'] = {
                    'semilandmarks_transformed': pca_v2_transformed,
                    'landmarks_transformed': pca_v2_deca_transformed,
                    'time': pca_v2_time,
                    'semi_stats': pca_v2_semi_stats,
                    'landmark_stats': pca_v2_landmark_stats,
                    'success': True
                }

            except Exception as e:
                print(f"  ❌ PCA-CPD-V2 failed: {e}")
                results['pca_v2'] = {'success': False, 'error': str(e)}

            # Traditional CPD
            print("\n--- Traditional CPD ---")
            start_time = time.time()
            try:
                traditional_reg = DeformableRegistration(
                    X=target_semilandmarks_downsampled,  # Target (fixed)
                    Y=skull_source,  # Source (moving)
                    alpha=2,
                    beta=1,
                    tolerance=0.001,
                    w=0.1,
                    max_iterations=200
                )

                traditional_transformed, _ = traditional_reg.register()
                traditional_time = time.time() - start_time
                print(f"  ✅ Traditional CPD completed in {traditional_time:.2f} seconds")

                # Analyze Traditional transformation
                traditional_semi_stats = analyze_transformation_quality(
                    skull_source, traditional_transformed, target_semilandmarks_downsampled, "Traditional Semilandmark"
                )

                # Create TPS and apply to landmarks
                traditional_tps_transform = calculate_tps_transform(skull_source, traditional_transformed)
                traditional_deca_transformed = traditional_tps_transform(deca_mean_source)

                traditional_landmark_stats = analyze_transformation_quality(
                    deca_mean_source, traditional_deca_transformed, target_landmarks, "Traditional Landmark"
                )

                results['traditional'] = {
                    'semilandmarks_transformed': traditional_transformed,
                    'landmarks_transformed': traditional_deca_transformed,
                    'time': traditional_time,
                    'semi_stats': traditional_semi_stats,
                    'landmark_stats': traditional_landmark_stats,
                    'success': True
                }

            except Exception as e:
                print(f"  ❌ Traditional CPD failed: {e}")
                results['traditional'] = {'success': False, 'error': str(e)}

            # Create visualizations for this specimen
            if any(r.get('success', False) for r in results.values()):
                viz_data = {
                    "Target Landmarks": target_landmarks,
                    "Mean Semilandmarks": skull_source,
                    "Target Semilandmarks": target_semilandmarks_downsampled
                }

                if results.get('pca', {}).get('success', False):
                    viz_data["PCA Landmarks"] = results['pca']['landmarks_transformed']
                if results.get('pca_v2', {}).get('success', False):
                    viz_data["PCA-V2 Landmarks"] = results['pca_v2']['landmarks_transformed']
                if results.get('traditional', {}).get('success', False):
                    viz_data["Traditional Landmarks"] = results['traditional']['landmarks_transformed']

                detailed_visualization_step(
                    viz_data,
                    f"{specimen_name} - Registration Results",
                    os.path.join(specimen_dir, f"{specimen_name}_results.png"),
                    1
                )

            # Generate summary report for this specimen
            specimen_summary = {
                'specimen_name': specimen_name,
                'initial_landmark_rmse': float(initial_landmark_rmse),
                'downsampling_ratio': downsampling_ratio,
                'data_characteristics': {
                    'original_semilandmarks': len(target_semilandmarks_full),
                    'downsampled_semilandmarks': len(target_semilandmarks_downsampled),
                    'centroid_distance': float(centroid_distance),
                    'target_complexity': float(np.std(target_distances)),
                    'mean_complexity': float(np.std(mean_distances)),
                    'scale_ratio': (bounds_target / bounds_mean).tolist()
                },
                'methods': {}
            }

            # Add method results
            for method_name, method_results in results.items():
                if method_results.get('success', False):
                    specimen_summary['methods'][method_name] = {
                        'final_rmse': float(method_results['landmark_stats']['final_rmse']),
                        'improvement_pct': float(method_results['landmark_stats']['improvement_pct']),
                        'time': float(method_results['time']),
                        'displacement_stats': {
                            'mean': float(method_results['semi_stats']['displacement_mean']),
                            'std': float(method_results['semi_stats']['displacement_std']),
                            'max': float(method_results['semi_stats']['displacement_max'])
                        }
                    }
                else:
                    specimen_summary['methods'][method_name] = {
                        'success': False,
                        'error': method_results.get('error', 'Unknown error')
                    }

            # Save individual specimen report
            with open(os.path.join(specimen_dir, f"{specimen_name}_summary.json"), 'w') as f:
                json.dump(specimen_summary, f, indent=2)

            all_results[specimen_name] = specimen_summary

            print(f"\n✅ {specimen_name} analysis complete!")

        except Exception as e:
            print(f"❌ Failed to analyze {specimen_name}: {e}")
            continue

    # Generate comparative analysis across all specimens
    print(f"\n\n{'=' * 80}")
    print("COMPARATIVE ANALYSIS ACROSS NEGATIVE SAMPLES")
    print(f"{'=' * 80}")

    # Create comparison report
    comparison_report = {
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'specimens_analyzed': list(all_results.keys()),
        'total_specimens': len(all_results),
        'method_failure_analysis': {},
        'data_characteristic_patterns': {},
        'specimens': all_results
    }

    # Analyze failure patterns
    for method in ['pca', 'pca_v2', 'traditional']:
        failures = []
        successes = []

        for specimen_name, results in all_results.items():
            method_result = results.get('methods', {}).get(method, {})
            if method_result.get('success', True):  # Default to success if not explicitly failed
                if 'improvement_pct' in method_result:
                    if method_result['improvement_pct'] < 0:
                        failures.append((specimen_name, method_result['improvement_pct']))
                    else:
                        successes.append((specimen_name, method_result['improvement_pct']))
            else:
                failures.append((specimen_name, 'CRASHED'))

        comparison_report['method_failure_analysis'][method] = {
            'total_failures': len(failures),
            'total_successes': len(successes),
            'failure_specimens': failures,
            'success_specimens': successes
        }

    # Save comprehensive comparison report
    with open(os.path.join(output_dir, "negative_samples_comparison.json"), 'w') as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Results saved to: {output_dir}/")
    print(f"📈 Comparison report: {output_dir}/negative_samples_comparison.json")

    return comparison_report


if __name__ == "__main__":
    analyze_negative_samples()