# Audio Processing with Python Technical Notes
<!-- A rectangular diagram depicting an intermediate audio processing pipeline in Python, illustrating an audio file or live stream loaded into a Python script using libraries like `librosa`, `sounddevice`, or `pyaudio`, processed with advanced techniques (e.g., feature extraction, filtering, effects), integrated with basic machine learning preprocessing, and producing outputs like enhanced visualizations (e.g., spectrograms), extracted features (e.g., MFCCs, tempo), or modified audio, with arrows indicating the flow from input to processing to output. -->

## Quick Reference
- **Definition**: Intermediate audio processing with Python involves using Python libraries to perform advanced audio manipulation, feature extraction (e.g., MFCCs, beat tracking), and preprocessing for machine learning, enabling tasks like audio classification or effects application.
- **Key Use Cases**: Music analysis (e.g., beat detection), speech processing, audio effects (e.g., pitch shifting), and preparing audio data for machine learning models.
- **Prerequisites**: Familiarity with Python, NumPy, basic audio concepts (e.g., sampling rate, waveform), and introductory signal processing.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: Intermediate audio processing with Python leverages libraries to analyze, manipulate, and visualize audio, extracting complex features (e.g., chroma, tempo) and applying effects, often preparing data for machine learning applications.
- **Why**: Python’s ecosystem offers powerful tools for intermediate users to bridge signal processing and machine learning, enabling sophisticated audio tasks with accessible APIs.
- **Where**: Used in music recommendation systems, speech recognition, audio effect plugins, and research for tasks like genre classification or audio segmentation.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - Audio is processed as NumPy arrays, representing amplitude over time, with transformations into frequency or time-frequency domains (e.g., spectrograms) for analysis.
  - Feature extraction targets musically or acoustically meaningful properties like Mel-frequency cepstral coefficients (MFCCs), chroma, or beat locations, useful for analysis or machine learning.
  - Audio manipulation includes effects like filtering, pitch shifting, or time stretching, while preprocessing prepares features for models (e.g., normalization, augmentation).
- **Key Components**:
  - **Audio I/O**: Load files with `librosa.load` or stream audio with `pyaudio`/`sounddevice` for real-time processing.
  - **Feature Extraction**:
    - **MFCCs**: Timbral features via `librosa.feature.mfcc`.
    - **Chroma**: Harmonic content with `librosa.feature.chroma_stft`.
    - **Beat Tracking**: Rhythm analysis using `librosa.beat.beat_track`.
  - **Signal Processing**:
    - **Filtering**: Apply filters (e.g., low-pass) using `scipy.signal`.
    - **Spectrograms**: Frequency analysis with `librosa.stft` or `librosa.feature.melspectrogram`.
  - **Effects**: Pitch shifting or time stretching with `librosa.effects`.
  - **Preprocessing for ML**: Normalize features (e.g., `sklearn.preprocessing`) and augment data for robust models.
  - **Visualization**: Plot spectrograms, chroma, or waveforms with `librosa.display` and `matplotlib`.
- **Common Misconceptions**:
  - Misconception: All audio processing tasks require real-time performance.
    - Reality: Many tasks (e.g., feature extraction) are offline, but streaming is possible with optimization.
  - Misconception: Audio features are directly usable in ML models.
    - Reality: Features often need normalization, reshaping, or augmentation for effective training.

### Visual Architecture
```mermaid
graph TD
    A[Audio Input <br> (File or Stream)] --> B[Python Libraries <br> (librosa, pyaudio, scipy)]
    B --> C[Processing <br> (Features, Filters, Effects)]
    C --> D[ML Preprocessing <br> (Normalization, Augmentation)]
    C --> E[Output <br> (Spectrograms, Features, Audio)]
```
- **System Overview**: The diagram shows audio input processed by Python libraries for feature extraction, effects, or ML preprocessing, producing visualizations, features, or modified audio.
- **Component Relationships**: Input is transformed by processing, which supports ML preprocessing and generates diverse outputs.

## Implementation Details
### Intermediate Patterns
```python
# Example: Extract features, apply effects, and preprocess for ML with Python
import librosa
import librosa.display
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Load audio file
audio_path = "example.wav"  # Replace with your file or use librosa.ex('trumpet')
y, sr = librosa.load(audio_path, sr=22050)

# Extract features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Timbral features
chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Harmonic features
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)  # Rhythm
beat_times = librosa.frames_to_time(beats, sr=sr)
print(f"Tempo: {tempo:.2f} BPM, First 5 beat times: {beat_times[:5].round(2)}")

# Apply low-pass filter
b, a = signal.butter(4, 1000 / (sr / 2), btype="low")  # 1 kHz cutoff
y_filtered = signal.filtfilt(b, a, y)

# Apply pitch shift
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)  # Up 2 semitones

# Preprocess MFCCs for ML
scaler = StandardScaler()
mfcc_normalized = scaler.fit_transform(mfcc.T).T  # Normalize across time
print("Normalized MFCC shape:", mfcc_normalized.shape)

# Play original and modified audio
print("Playing original audio...")
sd.play(y, sr)
sd.wait()
print("Playing filtered audio...")
sd.play(y_filtered, sr)
sd.wait()
print("Playing pitch-shifted audio...")
sd.play(y_shifted, sr)
sd.wait()

# Visualize
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.vlines(beat_times, -1, 1, color="r", linestyle="--", label="Beats")
plt.title("Waveform with Beat Markers")
plt.legend()

plt.subplot(3, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.ylim(0, 5000)

plt.subplot(3, 1, 3)
librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma")
plt.colorbar()
plt.title("Chroma Features")
plt.tight_layout()
plt.show()

# Save modified audio
librosa.output.write("filtered.wav", y_filtered, sr)
librosa.output.write("pitch_shifted.wav", y_shifted, sr)
```
- **Step-by-Step Setup**:
  1. **Install Dependencies**:
     - Install Python (download from python.org).
     - Install libraries: `pip install librosa sounddevice numpy matplotlib scipy scikit-learn`.
     - Install `ffmpeg` for MP3 support: `conda install ffmpeg` or `sudo apt-get install ffmpeg`.
  2. **Prepare Audio**: Use a WAV file (e.g., `example.wav`) or Librosa’s example: `audio_path = librosa.ex('trumpet')`.
  3. **Save Code**: Save as `audio_processing_intermediate.py`.
  4. **Run**: Execute with `python audio_processing_intermediate.py` (ensure speakers are connected).
- **Code Walkthrough**:
  - Loads audio with `librosa.load` at 22.05 kHz.
  - Extracts MFCCs (`librosa.feature.mfcc`), chroma (`librosa.feature.chroma_stft`), and beat times (`librosa.beat.beat_track`).
  - Applies a low-pass filter using `scipy.signal.butter` and `filtfilt`.
  - Performs pitch shifting with `librosa.effects.pitch_shift`.
  - Normalizes MFCCs with `sklearn.preprocessing.StandardScaler` for ML.
  - Plays original, filtered, and pitch-shifted audio with `sounddevice`.
  - Visualizes waveform with beat markers, spectrogram, and chroma using `librosa.display`.
  - Saves modified audio files.
- **Common Pitfalls**:
  - Missing `ffmpeg` for non-WAV formats (install via `conda` or system package manager).
  - Audio device conflicts (check with `python -m sounddevice`).
  - Incorrect feature shapes for ML (ensure proper transposition).

## Real-World Applications
### Industry Examples
- **Use Case**: Speech emotion recognition.
  - Extracts MFCCs and chroma for training machine learning models.
- **Implementation Patterns**: Use `librosa` for feature extraction and `scikit-learn` for classification.
- **Success Metrics**: >75% classification accuracy, efficient processing.

### Hands-On Project
- **Project Goals**: Analyze and modify audio with feature extraction and effects.
- **Implementation Steps**:
  1. Run the code with a music or speech file.
  2. Verify beat times align with audio rhythm in the waveform plot.
  3. Compare original, filtered, and pitch-shifted audio playback.
  4. Check normalized MFCCs for ML compatibility (shape should be consistent).
- **Validation Methods**: Confirm visualizations reflect audio content; ensure filtered audio reduces high frequencies; verify pitch shift raises pitch audibly.

## Tools & Resources
### Essential Tools
- **Development Environment**: Python, Jupyter for interactive analysis.
- **Key Libraries**: `librosa` (analysis), `sounddevice` (I/O), `scipy` (signal processing), `numpy`, `matplotlib`, `scikit-learn`.
- **Testing Tools**: Audacity for audio verification, audio players (e.g., VLC).

### Learning Resources
- **Documentation**: Librosa docs (https://librosa.org/doc/), Scipy docs (https://docs.scipy.org/doc/scipy/), Sounddevice docs (https://python-sounddevice.readthedocs.io).
- **Tutorials**: Audio processing with Python (https://towardsdatascience.com/audio-processing-in-python-using-librosa-95d0f08d8567).
- **Community Resources**: Stack Overflow, r/DSP, Librosa GitHub (https://github.com/librosa/librosa).

## References
- Librosa documentation: https://librosa.org/doc/
- Scipy signal processing: https://docs.scipy.org/doc/scipy/reference/signal.html
- Audio processing tutorial: https://towardsdatascience.com/audio-processing-in-python-using-librosa-95d0f08d8567
- Digital signal processing basics: https://en.wikipedia.org/wiki/Digital_signal_processing

## Appendix
- **Glossary**:
  - **MFCC**: Mel-frequency cepstral coefficients for timbre analysis.
  - **Chroma**: Harmonic pitch class representation.
  - **Low-Pass Filter**: Removes high-frequency components.
- **Setup Guides**:
  - Install libraries: `pip install librosa sounddevice numpy matplotlib scipy scikit-learn`.
  - Install `ffmpeg`: `conda install ffmpeg` or `sudo apt-get install ffmpeg`.
- **Code Templates**:
  - Real-time streaming: Use `pyaudio` for live audio processing.
  - Onset detection: Use `librosa.onset.onset_detect` for event analysis.