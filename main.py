import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    read_signal,
    preprocess_signal,
    feature_extraction,
)


def main():
    path = "data/train/s1.txt"
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return

    sig = read_signal(path)
    if sig is None:
        print(f"Failed to read signal from {path}")
        return

    target_fs = 250

    # Preprocessing pipeline
    sig, segments = preprocess_signal(sig)
    
    if not segments or len(segments) == 0:
        print("Warning: no segments found")
        return

    # Get first segment
    first_segment = segments[0]
    
    # Extract features
    feats = feature_extraction(first_segment)
    print(f"Feature extraction: {len(feats)} features extracted from first segment")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: First segment after preprocessing
    ax = axes[0]
    time_seg = np.arange(len(first_segment)) / target_fs
    ax.plot(time_seg, first_segment, linewidth=1.2, color='blue')
    ax.set_title(f"First Segment After Preprocessing ({len(segments)} total segments)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Plot 2: Feature extraction
    ax = axes[1]
    ax.stem(range(len(feats)), feats, linefmt='b-', markerfmt='bo', basefmt='r-')
    ax.set_title(f"Feature Extraction (DCT coefficients, {len(feats)} features)")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Coefficient Value")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
