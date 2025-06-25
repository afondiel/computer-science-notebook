# Audio Processing with Python Technical Notes
<!-- A rectangular diagram illustrating the audio processing pipeline in Python, showing an audio file (e.g., WAV, MP3) loaded into a Python script using libraries like `librosa` or `sounddevice`, processed to perform tasks (e.g., playback, visualization, simple feature extraction), and producing outputs like plots (e.g., waveform) or modified audio, with arrows indicating the flow from audio input to processing to output. -->

## Quick Reference
- **Definition**: Audio processing with Python involves using Python libraries to load, manipulate, analyze, and visualize audio data for tasks like playback, recording, or basic feature extraction (e.g., volume, pitch).
- **Key Use Cases**: Playing/recording audio, visualizing waveforms, extracting basic audio features, and prototyping audio applications.
- **Prerequisites**: Basic Python knowledge and familiarity with installing Python packages.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: Audio processing with Python uses libraries to work with audio files or streams, enabling tasks like playing sound, recording audio, visualizing waveforms, or extracting simple features.
- **Why**: Python’s simplicity and rich ecosystem of audio libraries make it ideal for beginners to explore audio processing without needing deep signal processing knowledge.
- **Where**: Used in music analysis, speech processing, sound design, and educational projects for audio-related tasks.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - Audio is represented digitally as a sequence of amplitude samples, typically stored in files (e.g., WAV, MP3) or captured live via a microphone.
  - Python libraries convert audio into NumPy arrays for processing, where each value represents the amplitude at a given time, sampled at a rate (e.g., 44.1 kHz).
  - Common tasks include playback, recording, visualization (e.g., waveform plots), and basic analysis (e.g., detecting loudness).
- **Key Components**:
  - **Audio Loading**: Libraries like `librosa` load audio files into arrays, specifying sampling rates (e.g., 22.05 kHz).
  - **Playback and Recording**: Libraries like `sounddevice` play or record audio through the computer’s sound card.
  - **Visualization**: Plot audio data as waveforms or spectrograms using `matplotlib` and `librosa.display`.
  - **Basic Feature Extraction**: Compute simple features like amplitude (loudness) or duration using `numpy` or `librosa`.
  - **File Formats**: Common formats include WAV (uncompressed, high quality) and MP3 (compressed, smaller size).
- **Common Misconceptions**:
  - Misconception: Audio processing requires advanced math.
    - Reality: Beginners can use high-level library functions to achieve results without complex math.
  - Misconception: Python is too slow for audio processing.
    - Reality: Libraries use optimized C backends (e.g., `numpy`, `librosa`) for efficient processing.

### Visual Architecture
```mermaid
graph TD
    A[Audio Input <br> (e.g., WAV, Mic)] --> B[Python Library <br> (librosa, sounddevice)]
    B --> C[Processing <br> (Playback, Visualization)]
    C --> D[Output <br> (Waveform Plot, Audio)]
```
- **System Overview**: The diagram shows audio input loaded or captured by Python libraries, processed for playback or visualization, and producing outputs like plots or sound.
- **Component Relationships**: Input is processed by libraries, which generate meaningful outputs for analysis or playback.

## Implementation Details
### Basic Implementation
```python
# Example: Load, play, visualize, and analyze an audio file with Python
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_path = "example.wav"  # Replace with your WAV file or use librosa.ex('trumpet')
y, sr = librosa.load(audio_path, sr=22050)  # Load at 22.05 kHz

# Play audio
print("Playing audio...")
sd.play(y, sr)
sd.wait()  # Wait until playback finishes

# Calculate basic features
duration = librosa.get_duration(y=y, sr=sr)
rms = np.sqrt(np.mean(y**2))  # Root mean square (loudness)
print(f"Duration: {duration:.2f} seconds")
print(f"RMS (loudness): {rms:.4f}")

# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Record audio (5 seconds)
print("Recording audio for 5 seconds...")
recording = sd.rec(int(5 * sr), samplerate=sr, channels=1)
sd.wait()  # Wait until recording finishes
print("Recording complete")

# Save recording
librosa.output.write("recording.wav", recording.T[0], sr)  # Save as WAV
```
- **Step-by-Step Setup**:
  1. **Install Dependencies**:
     - Install Python (download from python.org).
     - Install libraries: `pip install librosa sounddevice numpy matplotlib`.
     - Install `ffmpeg` for MP3 support: `conda install ffmpeg` or `sudo apt-get install ffmpeg` (Ubuntu/Debian).
  2. **Prepare Audio**: Place a WAV file (e.g., `example.wav`) in your directory or use `audio_path = librosa.ex('trumpet')`.
  3. **Save Code**: Save as `audio_processing_beginner.py`.
  4. **Run**: Execute with `python audio_processing_beginner.py` (ensure speakers/microphone are connected).
- **Code Walkthrough**:
  - Loads audio with `librosa.load` into a NumPy array (`y`) and sampling rate (`sr`).
  - Plays audio using `sounddevice.play`.
  - Computes duration (`librosa.get_duration`) and RMS loudness (`numpy`).
  - Plots waveform with `librosa.display.waveshow`.
  - Records 5 seconds of audio with `sounddevice.rec` and saves it using `librosa.output.write`.
- **Common Pitfalls**:
  - Missing `ffmpeg` for non-WAV formats (install via `conda` or system package manager).
  - Incorrect audio device setup (check with `python -m sounddevice`).
  - Large audio files causing memory issues (use `duration` parameter in `librosa.load`).

## Real-World Applications
### Industry Examples
- **Use Case**: Audio prototyping for apps.
  - Play and record audio for user interfaces or voice assistants.
- **Implementation Patterns**: Use `sounddevice` for playback/recording and `librosa` for visualization.
- **Success Metrics**: Clear audio output, reliable recording, user-friendly prototypes.

### Hands-On Project
- **Project Goals**: Load, play, record, and visualize audio.
- **Implementation Steps**:
  1. Run the code with a WAV file or Librosa’s example audio.
  2. Listen to playback and check waveform plot for amplitude changes.
  3. Record your voice for 5 seconds and play back `recording.wav`.
  4. Verify RMS reflects audio loudness (louder sounds yield higher RMS).
- **Validation Methods**: Confirm playback/recording quality; ensure waveform matches audio dynamics.

## Tools & Resources
### Essential Tools
- **Development Environment**: Python, Jupyter notebooks for interactive coding.
- **Key Libraries**: `librosa` (analysis), `sounddevice` (playback/recording), `numpy` (math), `matplotlib` (plotting).
- **Testing Tools**: Audio players (e.g., VLC), Audacity for audio inspection.

### Learning Resources
- **Documentation**: Librosa docs (https://librosa.org/doc/), Sounddevice docs (https://python-sounddevice.readthedocs.io).
- **Tutorials**: Audio processing with Python (https://realpython.com/playing-and-recording-sound-python/).
- **Community Resources**: Stack Overflow, r/Python, Librosa GitHub (https://github.com/librosa/librosa).

## References
- Librosa documentation: https://librosa.org/doc/
- Sounddevice documentation: https://python-sounddevice.readthedocs.io
- Python audio processing tutorial: https://realpython.com/playing-and-recording-sound-python/
- Audio basics: https://en.wikipedia.org/wiki/Digital_audio

## Appendix
- **Glossary**:
  - **Sampling Rate**: Samples per second (e.g., 44.1 kHz for CD quality).
  - **Waveform**: Visual representation of audio amplitude over time.
  - **RMS**: Root mean square, a measure of audio loudness.
- **Setup Guides**:
  - Install libraries: `pip install librosa sounddevice numpy matplotlib`.
  - Install `ffmpeg`: `conda install ffmpeg` or `sudo apt-get install ffmpeg`.
- **Code Templates**:
  - Simple pitch analysis: Use `librosa.piptrack` for pitch extraction.
  - Volume adjustment: Scale `y` array with `numpy` before playback.