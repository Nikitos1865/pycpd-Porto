import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from pathlib import Path


def load_landmarks(json_path):
    """Load landmark coordinates from a .mrk.json file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        return np.array([cp['position'] for cp in data['markups'][0]['controlPoints']], dtype=float)


def compute_procrustes_distance(A, B):
    try:
        _, _, disparity = procrustes(A, B)
        return disparity
    except Exception as e:
        print(f"⚠️ Procrustes failed: {e}")
        return np.nan


def compute_procrustes_for_aligned_LMs(mean_path, samples_dir):
    print(f"📂 Loading decamean landmarks from: {mean_path}")
    mean_landmarks = load_landmarks(mean_path)

    results = []

    for file in os.listdir(samples_dir):
        if file.endswith('.mrk.json') and not file.startswith('._'):
            sample_path = os.path.join(samples_dir, file)
            try:
                sample_landmarks = load_landmarks(sample_path)
                distance = compute_procrustes_distance(mean_landmarks, sample_landmarks)
                results.append({
                    'specimen_name': Path(file).stem,
                    'procrustes_distance': distance,
                })
            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")

    return pd.DataFrame(results)


def plot_procrustes_histogram(df, out_path):
    plt.figure(figsize=(10, 6))
    plt.hist(df['procrustes_distance'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title('Procrustes Distance Distribution')
    plt.xlabel('Procrustes Distance')
    plt.ylabel('Frequency')

    mean = df['procrustes_distance'].mean()
    median = df['procrustes_distance'].median()
    std = df['procrustes_distance'].std()
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    print(f"📊 Histogram saved to {out_path}")


def print_procrustes_stats(df):
    print("\n📈 Procrustes Distance Statistics:")
    print(f"Mean:    {df['procrustes_distance'].mean():.6f}")
    print(f"Median:  {df['procrustes_distance'].median():.6f}")
    print(f"Std:     {df['procrustes_distance'].std():.6f}")
    print(f"Min:     {df['procrustes_distance'].min():.6f}")
    print(f"Max:     {df['procrustes_distance'].max():.6f}")

    print("\n🔺 Top 5 Highest Distances:")
    print(df.nlargest(5, 'procrustes_distance')[['specimen_name', 'procrustes_distance']])

    print("\n🔻 Top 5 Lowest Distances:")
    print(df.nsmallest(5, 'procrustes_distance')[['specimen_name', 'procrustes_distance']])


def main():
    print("🚀 Starting Procrustes distance analysis...")

    mean_path = "../data/mean/decaMeanModel.mrk.json"
    samples_dir = "../data/aligned_LMs"
    output_dir = "procrustes_analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    df = compute_procrustes_for_aligned_LMs(mean_path, samples_dir)

    if df.empty:
        print("❌ No data processed. Exiting.")
        return

    # Save raw results
    csv_path = os.path.join(output_dir, "procrustes_distances.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Procrustes distances saved to {csv_path}")

    # Plot
    plot_path = os.path.join(output_dir, "procrustes_histogram.png")
    plot_procrustes_histogram(df, plot_path)

    # Stats
    print_procrustes_stats(df)


if __name__ == "__main__":
    main()
