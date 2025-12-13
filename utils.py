import os
import numpy as np
from scipy import signal as sp_signal
from scipy import fftpack as sp_fft
from sklearn.ensemble import RandomForestClassifier
fs = 1000


def read_signal(path):
    if os.path.isfile(path):
        data = np.loadtxt(path, dtype=np.float64)
        return data
    else:
        print("Error: File Not Exist")


def train():
    train_data = load_train()
    X, y = [], []
    for label, signal in train_data.items():
        _, segments = preprocess_signal(signal)
        for seg in segments:
            feats = feature_extraction(seg)[:40]
            if len(feats) < 40:
                feats = np.pad(feats, (0, 40 - len(feats)))
            X.append(feats)
            y.append(label)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf


def test(clf):
    test_data = load_test()
    predictions = []
    for signal in test_data:
        _, segments = preprocess_signal(signal)
        seg_preds = []
        for seg in segments:
            feats = feature_extraction(seg)[:40]
            if len(feats) < 40:
                feats = np.pad(feats, (0, 40 - len(feats)))
            pred = clf.predict([feats])[0]
            seg_preds.append(pred)
        from collections import Counter
        majority = Counter(seg_preds).most_common(1)[0][0]
        predictions.append(majority)
    return predictions


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


def bandpass_filter(signal, lowcut, highcut, fs=1000, order=4):
    if lowcut <= 0 or highcut >= fs / 2:
        raise ValueError(
            f"Invalid cutoffs: low={lowcut}, high={highcut}, nyquist={fs/2}"
        )
    if lowcut >= highcut:
        raise ValueError("Bandpass lowcut must be < highcut")

    b, a = sp_signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    result_signal = sp_signal.filtfilt(b, a, signal)
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


def preprocess_signal(signal, fs=1000, lowcut=1, highcut=40):
    target_fs = 250
    signal = mean_removal(signal)
    signal = resample_signal(signal, target_fs, fs)
    signal = bandpass_filter(signal, lowcut, highcut, target_fs)
    signal = normalize_signal(signal)
    segments = segment_ecg(signal, target_fs)
    return signal, segments


def autocorrelation(signal):
    result = sp_signal.correlate(signal, signal,mode='full')
    result = result[result.size // 2 :]
    result /= np.max(result)
    return result


def select_sign_coff(result_signal, threshold=0.05):
    indices = np.where(result_signal > threshold)[0]
    return result_signal[indices]

def feature_extraction(signal, threshold=1e-5):
    result = autocorrelation(signal)
    result = select_sign_coff(result)
    features = sp_fft.dct(result, norm='ortho')
    features = features[np.abs(features) > threshold]
    return features
