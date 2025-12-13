import os
import numpy as np
from scipy import signal as sp_signal

fs = 1000


def read_signal(path):
    if os.path.isfile(path):
        data = np.loadtxt(path, dtype=np.float64)
        return data
    else:
        print("Error: File Not Exist")


def train():
    pass


def test():
    pass


def load_train():
    full_path = "data/train"
    train_data = {}
    for file_name in os.listdir(full_path):
        label = "person" + os.path.splitext(file_name)[0][1]
        data = read_signal(os.path.join(full_path, file_name))
        train_data[label] = data
    return train_data


def load_test():
    full_path = "data/test"
    files = os.listdir(full_path)
    test_data = np.empty(len(files), dtype=object)
    for i, file_name in enumerate(files):
        data = read_signal(os.path.join(full_path, file_name))
        test_data[i] = data
    return test_data


def mean_removal(signal):
    mean = np.mean(signal)
    return signal - mean


def bandpass_filter(signal, lowcut, highcut, fs=1000, N=201):
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0 or high >= 1:
        raise ValueError(
            f"Invalid cutoffs: low={lowcut}, high={highcut}, nyquist={nyquist}"
        )
    if low >= high:
        raise ValueError("Bandpass lowcut must be < highcut")

    if N % 2 == 0:
        N += 1

    b = sp_signal.firwin(
        N,
        [lowcut, highcut],
        pass_zero=False,
        fs=fs,
        window="hamming",
    )
    result_signal = sp_signal.filtfilt(b, [1.0], signal)
    return result_signal


def normalize_signal(signal):
    std = np.std(signal)
    if std == 0:
        return signal
    return signal / std


def resample_signal(signal, target_fs, fs):
    samples = int(len(signal) * target_fs / fs)
    return sp_signal.resample(signal, samples)


def segment_ecg(signal, fs, beats_per_segment=4, min_rr_interval=0.4):
    distance = int(min_rr_interval * fs)
    height = np.mean(signal) + 0.5 * np.std(signal)
    r_peaks, _ = sp_signal.find_peaks(signal, distance=distance, height=height)

    segments = []
    for i in range(0, len(r_peaks) - beats_per_segment + 1, beats_per_segment):
        start = r_peaks[i]
        end = r_peaks[i + beats_per_segment - 1]
        segments.append(signal[start:end])

    return segments


def preprocess_signal(signal, fs=1000, lowcut=0.5, highcut=40):
    signal = mean_removal(signal)
    signal = bandpass_filter(signal, lowcut, highcut, fs)
    signal = normalize_signal(signal)
    signal = resample_signal(signal, 250, fs)
    segments = segment_ecg(signal, fs)
    return signal, segments
