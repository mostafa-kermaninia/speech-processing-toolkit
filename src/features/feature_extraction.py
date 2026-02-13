import librosa
import numpy as np
import tensorflow as tf
from src.config import N_MFCC, N_MELS, MAX_DIMS

def compute_f0(y, sr, fmin=50, fmax=500):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    return f0

def compute_mfcc(y, sr, n_mfcc=N_MFCC):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def compute_spectral_centroid(y, sr):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return spectral_centroid

def compute_spectral_flux(y, sr):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    stft = np.abs(librosa.stft(y=y))
    spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    return spectral_flux

def compute_spectral_bandwidth(y, sr):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return spectral_bandwidth

def compute_spectral_contrast(y, sr, n_bands=6):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands)
    return spectral_contrast

def compute_zcr(y, sr):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    return zcr

def compute_jitter(y, sr):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    f0 = librosa.yin(y=y, fmin=50, fmax=sr//2, sr=sr)
    jitter = np.mean(np.abs(np.diff(f0)) / f0[:-1])
    return jitter

def compute_shimmer(y, sr):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    frame_length = int(sr * 0.03)
    hop_length = frame_length // 2
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    shimmer = np.mean(np.abs(np.diff(energy)) / energy[:-1])
    return shimmer

def compute_energy(y, sr):
    energy = np.sum(np.square(y))
    return energy

def compute_log_mel_spectrogram(y, sr, n_mels=N_MELS):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def compute_chroma_features(y, sr, n_chroma=12):
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    return chroma

def extract_features(y, sr, file_name=None):
    raw_features = {
        "fundamental_frequency": compute_f0(y, sr),
        "mfcc": compute_mfcc(y, sr),
        "spectral_centroid": compute_spectral_centroid(y, sr),
        "spectral_flux": compute_spectral_flux(y, sr),
        "spectral_bandwidth": compute_spectral_bandwidth(y, sr),
        "zero_crossing_rate": compute_zcr(y, sr),
        "jitter": compute_jitter(y, sr),
        "shimmer": compute_shimmer(y, sr),
        "energy": compute_energy(y, sr),
        "log_mel_spectrogram": compute_log_mel_spectrogram(y, sr),
        "chroma_features": compute_chroma_features(y, sr),
        "spectral_contrast": compute_spectral_contrast(y, sr),
    }

    if file_name:
        raw_features["file_name"] = file_name

    reduced_features = {}

    for key, value in raw_features.items():
        if isinstance(value, np.ndarray) or isinstance(value, list):
            value = np.array(value)
            if value.ndim == 1:
                reduced_features[f"{key}_mean"] = np.mean(value)
                reduced_features[f"{key}_std"] = np.std(value)
            elif value.ndim == 2:
                mean_values = np.mean(value, axis=1)
                selected_dims = min(MAX_DIMS, mean_values.shape[0])
                selected_values = mean_values[:selected_dims]
                for i, v in enumerate(selected_values):
                    reduced_features[f"{key}_{i}"] = v
        else:
             reduced_features[key] = value

    return reduced_features
