import os
import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 48000  # Target sample rate for resampling
NUM_CHANNELS = 2  # Number of channels for stereo
MAX_DURATION = 11.9  # Maximum duration in seconds

def load_audio(file_path, target_sr=TARGET_SR):
    y, sr = librosa.load(file_path, sr=None)
    """ if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr """
    return y, sr

def save_audio(file_path, y, sr):
    if y.ndim == 1:  # Convert mono to stereo
        y = np.tile(y[:, np.newaxis], (1, NUM_CHANNELS))
    sf.write(file_path, y, sr)

def pad_or_trim(y, target_length):
    if len(y) < target_length:
        # Pad with zeros if shorter
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    elif len(y) > target_length:
        # Trim if longer
        y = y[:target_length]
    return y

def match_audio_lengths(reference_path, stimuli_paths, target_sr=TARGET_SR):
    ref_audio, sr = load_audio(reference_path, target_sr)
    max_length = min(len(ref_audio), int(MAX_DURATION * sr))

    stimuli_audios = []
    for stimulus_path in stimuli_paths:
        stim_audio, _ = load_audio(stimulus_path, target_sr)
        stimuli_audios.append(stim_audio)
        max_length = min(max(max_length, len(stim_audio)), int(MAX_DURATION * sr))

    ref_audio = pad_or_trim(ref_audio, max_length)
    save_audio(reference_path, ref_audio, sr)

    for stimulus_path, stim_audio in zip(stimuli_paths, stimuli_audios):
        adjusted_audio = pad_or_trim(stim_audio, max_length)
        save_audio(stimulus_path, adjusted_audio, sr)

# This is so hardcoded it hurts me
test_groups = [
    {
        "reference": "AUDIO_RESULTS/processed/SPF05_AviaoVocal.wav",
        "stimuli": [
            "AUDIO_RESULTS/processed/SPF05_AviaoWhisperd.wav",
            "AUDIO_RESULTS/unsegmented/SPF05_Aviao_1_frameFiltering.wav",
            "AUDIO_RESULTS/unsegmented/SPF05_Aviao_2_pulseFiltering.wav",
            "AUDIO_RESULTS/segmented/SPF05_Aviao_3_segmentedF0.wav"
        ]
    },
    {
        "reference": "AUDIO_RESULTS/processed/SPF12_TiagoVocal.wav",
        "stimuli": [
            "AUDIO_RESULTS/processed/SPF12_TiagoWhisperd.wav",
            "AUDIO_RESULTS/unsegmented/SPF12_Tiago_1_frameFiltering.wav",
            "AUDIO_RESULTS/unsegmented/SPF12_Tiago_2_pulseFiltering.wav",
            "AUDIO_RESULTS/segmented/SPF12_Tiago_3_segmentedF0.wav"
        ]
    },
    {
        "reference": "AUDIO_RESULTS/processed/SPM14_TiagoVocal.wav",
        "stimuli": [
            "AUDIO_RESULTS/processed/SPM14_TiagoWhisperd.wav",
            "AUDIO_RESULTS/unsegmented/SPM14_Tiago_1_frameFiltering.wav",
            "AUDIO_RESULTS/unsegmented/SPM14_Tiago_2_pulseFiltering.wav",
            "AUDIO_RESULTS/segmented/SPM14_Tiago_3_segmentedF0.wav"
        ]
    },
    {
        "reference": "AUDIO_RESULTS/processed/SPM18_AviaoVocal.wav",
        "stimuli": [
            "AUDIO_RESULTS/processed/SPM18_AviaoWhisperd.wav",
            "AUDIO_RESULTS/unsegmented/SPM18_Aviao_1_frameFiltering.wav",
            "AUDIO_RESULTS/unsegmented/SPM18_Aviao_2_pulseFiltering.wav",
            "AUDIO_RESULTS/segmented/SPM18_Aviao_3_segmentedF0.wav"
        ]
    },
    {
        "reference": "AUDIO_RESULTS/vowel/source.wav",
        "stimuli": [
            "AUDIO_RESULTS/vowel/f0.wav",
            "AUDIO_RESULTS/vowel/constant.wav"
        ]
    }
]

# Adjust lengths for each reference and its corresponding stimuli
for group in test_groups:
    match_audio_lengths(group["reference"], group["stimuli"])

print("Audio processing complete. All files have been resampled to 48kHz, converted to stereo, and adjusted to the same length as the longest file in their respective groups.")
