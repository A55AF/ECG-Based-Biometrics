# ECG-Based Subject Classification Using DSP and kNN

## Project Overview

This project implements a complete digital signal processing (DSP) pipeline for electrocardiogram (ECG) signal analysis and subject identification using the k-Nearest Neighbors (kNN) classifier. The goal is to classify individuals based on their unique ECG characteristics, demonstrating core DSP concepts including filtering, feature extraction, and pattern recognition.

**Dataset**: PTB ECG Database  
**Sampling Rate**: 1000 Hz  
**Subjects**: 3 individuals (s1, s2, s3)  
**Train/Test Split**: Separate recordings per subject in `data/train/` and `data/test/`

---

## Project Goals

1. **Preprocess ECG signals**: Remove noise (baseline wander, mains hum) and smooth signals via filtering.
2. **Detect cardiac features**: Identify R-peaks and delineate P/QRS/T waves for beat-level analysis.
3. **Extract discriminative features**: Compute intervals (RR, PR, QRS, QT), heart rate variability (HRV), morphology (amplitudes, slopes), and waveform patterns.
4. **Classify subjects**: Train and evaluate a kNN model to distinguish between individuals.
5. **Report results**: Document accuracy, confusion matrix, and feature importance.

---

## Project Structure

```
ECG/
├── main.py              # Main pipeline: load, preprocess, detect, extract, classify
├── utils.py             # Core DSP and feature extraction functions
├── test.py              # Unit tests for utils functions
├── README.md            # This file
├── data/
│   ├── train/
│   │   ├── s1.txt       # Subject 1 training ECG signal (~23k samples)
│   │   ├── s2.txt       # Subject 2 training ECG signal
│   │   └── s3.txt       # Subject 3 training ECG signal
│   └── test/
│       ├── s1.txt       # Subject 1 test ECG signal
│       ├── s2.txt       # Subject 2 test ECG signal
│       └── s3.txt       # Subject 3 test ECG signal
└── testing/             # Output directory for metrics and model artifacts
```
