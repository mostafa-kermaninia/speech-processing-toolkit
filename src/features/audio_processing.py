import librosa
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter

def spectral_subtraction_with_dynamic_noise(y, sr, max_noise_duration=1.0, silence_threshold=-40):
    """Extracts silence-based noise and applies spectral subtraction."""
    intervals = librosa.effects.split(y, top_db=abs(silence_threshold))
    noise_segments = []

    for start, end in intervals:
        if end - start >= sr * 0.1:  # At least 100ms
            noise_segments.append(y[start:end])
            if sum(len(seg) for seg in noise_segments) >= sr * max_noise_duration:
                break

    if noise_segments:
        noise_sample = np.concatenate(noise_segments[:min(int(sr * max_noise_duration),len(noise_segments))])
    else:
        noise_sample = np.zeros(int(sr * max_noise_duration))

    denoised_signal = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return denoised_signal

def bandpass_filter(signal, lowcut, highcut, sr, order=6):
    """Applies a bandpass filter to the signal."""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def resample_audio(y, sr, target_sr=44100):
    """Resamples the audio to the target sample rate."""
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)

def trim_audio(y, top_db=20):
    """Trims silence from the audio."""
    trimmed_signal, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed_signal

def normalize_audio(y):
    """Normalizes the audio amplitude."""
    return librosa.util.normalize(y)

def trim_audio_to_spectral_centroid(file, needed_dur=30):
    """Trims audio centered around the spectral centroid."""
    y, sr = librosa.load(file, sr=None)
    duration = len(y) / sr

    if duration < needed_dur:
        return y, sr

    centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(centroid_frames, sr=sr)
    centroid_time = times[np.argmax(centroid_frames)]
    centroid_index = int(centroid_time * sr)

    segment_length = needed_dur * sr
    half_segment = segment_length // 2

    start_idx = max(0, centroid_index - half_segment)
    end_idx = min(len(y), centroid_index + half_segment)

    if start_idx == 0:
        end_idx = min(len(y), end_idx + abs(centroid_index - half_segment))
    if end_idx == len(y):
        start_idx = max(0, start_idx - abs(centroid_index + half_segment - len(y)))

    y_trimmed = y[int(start_idx):int(end_idx)]
    return y_trimmed, sr

def preprocess_audio_file(file, target_sr=44100):
    """Runs the full preprocessing pipeline on a single file."""
    y, sr = trim_audio_to_spectral_centroid(file, needed_dur=30)
    y_denoised = spectral_subtraction_with_dynamic_noise(y, sr)
    filtered_y = bandpass_filter(y_denoised, 80, 5000, sr)
    resampled_y = resample_audio(filtered_y, sr, target_sr=target_sr)
    trimmed_y = trim_audio(resampled_y, 25)
    normalized_y = normalize_audio(trimmed_y)
    return normalized_y, target_sr
