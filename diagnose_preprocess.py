import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

from utils import read_signal

# Reproduce each step manually to diagnose
def manual_mean_removal(signal):
    return signal - np.mean(signal)

def manual_bandpass_iir(signal, lowcut, highcut, fs, order=4):
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sp_signal.butter(order, [low, high], btype='band')
    return sp_signal.filtfilt(b, a, signal)

def manual_resample(signal, target_fs, fs):
    samples = int(len(signal) * target_fs / fs)
    return sp_signal.resample(signal, samples)

def manual_normalize(signal):
    std = np.std(signal)
    if std == 0:
        return signal
    return signal / std

def detect_peaks_simple(signal, fs, min_rr=0.4):
    distance = int(min_rr * fs)
    # Use prominence for more robust detection
    prominence = 0.8 * np.std(signal)
    peaks, _ = sp_signal.find_peaks(signal, distance=distance, prominence=prominence)
    return peaks

def plot_step(ax, signal, fs, title, peaks=None):
    time = np.arange(len(signal)) / fs
    ax.plot(time, signal, label=title, linewidth=0.8)
    if peaks is not None and len(peaks) > 0:
        ax.plot(time[peaks], signal[peaks], 'rx', markersize=8, label=f'{len(peaks)} R-peaks')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

def main():
    path = "data/train/s2.txt"
    print(f"Loading: {path}")
    sig = read_signal(path)
    if sig is None:
        print("Failed to read signal")
        return
    
    fs = 1000
    lowcut = 1.0
    highcut = 40.0
    target_fs = 250
    
    # Create figure with subplots for each step
    fig, axes = plt.subplots(7, 1, figsize=(14, 18))
    
    # Step 0: Raw signal
    raw = sig.copy()
    peaks_raw = detect_peaks_simple(raw, fs)
    plot_step(axes[0], raw, fs, "0. Raw Signal", peaks_raw)
    print(f"Raw: length={len(raw)}, mean={np.mean(raw):.3f}, std={np.std(raw):.3f}, peaks={len(peaks_raw)}")
    
    # Step 1: Mean removal
    step1 = manual_mean_removal(raw)
    peaks_1 = detect_peaks_simple(step1, fs)
    plot_step(axes[1], step1, fs, "1. After Mean Removal", peaks_1)
    print(f"Mean removal: length={len(step1)}, mean={np.mean(step1):.3f}, std={np.std(step1):.3f}, peaks={len(peaks_1)}")
    
    # Step 2: Resample (NEW ORDER: resample before bandpass)
    step2 = manual_resample(step1, target_fs, fs)
    peaks_2 = detect_peaks_simple(step2, target_fs)
    plot_step(axes[2], step2, target_fs, f"2. After Resample (1000Hz -> {target_fs}Hz)", peaks_2)
    print(f"Resample: length={len(step2)}, mean={np.mean(step2):.3f}, std={np.std(step2):.3f}, peaks={len(peaks_2)}")
    
    # Step 3: Bandpass filter (IIR order=2)
    step3 = manual_bandpass_iir(step2, lowcut, highcut, target_fs, order=2)
    peaks_3 = detect_peaks_simple(step3, target_fs)
    plot_step(axes[3], step3, target_fs, f"3. After Bandpass [{lowcut}-{highcut}Hz, IIR order=2]", peaks_3)
    print(f"Bandpass: length={len(step3)}, mean={np.mean(step3):.3f}, std={np.std(step3):.3f}, peaks={len(peaks_3)}")
    
    # Step 4: Median filter (suppress artifacts) - increased kernel
    step4 = sp_signal.medfilt(step3, kernel_size=9)
    peaks_4 = detect_peaks_simple(step4, target_fs)
    plot_step(axes[4], step4, target_fs, "4. After Median Filter (kernel=9)", peaks_4)
    print(f"Median filter (k=9): length={len(step4)}, mean={np.mean(step4):.3f}, std={np.std(step4):.3f}, peaks={len(peaks_4)}")
    
    # Step 5: Normalize
    step5 = manual_normalize(step4)
    peaks_5 = detect_peaks_simple(step5, target_fs)
    plot_step(axes[5], step5, target_fs, "5. After Normalization (z-score)", peaks_5)
    print(f"Normalize: length={len(step5)}, mean={np.mean(step5):.3f}, std={np.std(step5):.3f}, peaks={len(peaks_5)}")
    
    # Step 6: Zoom on first 2-3 beats of final signal
    if len(peaks_5) >= 2:
        zoom_start = max(0, peaks_5[0] - int(0.2 * target_fs))
        zoom_end = min(len(step5), peaks_5[min(2, len(peaks_5)-1)] + int(0.3 * target_fs))
        zoom_sig = step5[zoom_start:zoom_end]
        zoom_time = np.arange(len(zoom_sig)) / target_fs
        zoom_peaks = peaks_5[(peaks_5 >= zoom_start) & (peaks_5 < zoom_end)] - zoom_start
        
        axes[6].plot(zoom_time, zoom_sig, linewidth=1.2)
        axes[6].plot(zoom_time[zoom_peaks], zoom_sig[zoom_peaks], 'rx', markersize=10, label=f'{len(zoom_peaks)} peaks')
        axes[6].set_title("6. Zoomed: First 2-3 Beats (Final Signal)")
        axes[6].set_xlabel('Time (s)')
        axes[6].set_ylabel('Amplitude')
        axes[6].legend()
        axes[6].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/diagnose_preprocess.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved diagnostic plot to: {out_path}")
    plt.close()
    
    # Additional diagnostic: Check frequency content at each step
    print("\n--- FFT Analysis (top 5 frequency peaks) ---")
    for i, (step_sig, step_fs, step_name) in enumerate([
        (raw, fs, "Raw"),
        (step1, fs, "Mean Removal"),
        (step2, target_fs, "Resample"),
        (step3, target_fs, "Bandpass"),
        (step4, target_fs, "Median Filter"),
        (step5, target_fs, "Normalize"),
    ]):
        fft_mag = np.abs(np.fft.rfft(step_sig))
        fft_freqs = np.fft.rfftfreq(len(step_sig), 1.0 / step_fs)
        top_idx = np.argsort(-fft_mag)[:5]
        print(f"{step_name}:")
        for idx in top_idx:
            print(f"  {fft_freqs[idx]:.2f} Hz -> {fft_mag[idx]:.3g}")

if __name__ == '__main__':
    main()
