# Computer Audition Technical Notes
A rectangular diagram illustrating the computer audition process, showing an audio input (e.g., a sound wave from a microphone) processed through basic feature extraction (e.g., amplitude analysis), analyzed with a simple algorithm (e.g., threshold-based detection), producing an output (e.g., identifying a loud sound), with arrows indicating the flow from audio capture to processing to result.

## Quick Reference
- **Definition**: Computer audition is the field of enabling computers to analyze and interpret audio signals, such as detecting sounds, recognizing speech, or classifying noises, using computational techniques implemented in Python.
- **Key Use Cases**: Sound detection in smart devices, basic speech recognition, and environmental noise monitoring.
- **Prerequisites**: Basic understanding of Python programming and familiarity with audio as sound waves.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: Computer audition involves programming computers to "listen" to audio inputs and perform tasks like detecting a clap, identifying a siren, or transcribing simple speech using Python.
- **Why**: It enables applications like voice-controlled devices, audio-based alerts, and automated sound analysis in various environments.
- **Where**: Used in IoT devices, security systems, mobile apps, and research for tasks like audio event detection or basic voice interaction.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - Audio is captured as a digital signal, a sequence of numbers representing sound wave amplitude over time, typically sampled at rates like 16 kHz.
  - Computer audition processes these signals by extracting simple features (e.g., amplitude or energy) and applying algorithms to interpret them.
  - Basic algorithms, like threshold-based detection, identify sound events by comparing features to predefined values.
- **Key Components**:
  - **Audio Capture**: Recording sound using a microphone, converted to digital samples via an Analog-to-Digital Converter (ADC).
  - **Feature Extraction**: Computing basic properties like amplitude or root mean square (RMS) energy from audio samples.
  - **Analysis Algorithm**: A simple decision rule (e.g., if energy exceeds a threshold, detect a sound) to produce meaningful outputs.
- **Common Misconceptions**:
  - Misconception: Computer audition is only about speech recognition.
    - Reality: It includes non-speech sounds like environmental noises or musical notes.
  - Misconception: You need complex tools to start.
    - Reality: Python libraries like `pyaudio` make basic audition tasks accessible to beginners.

### Visual Architecture
```mermaid
graph TD
    A[Audio Input <br> (e.g., Sound Wave)] --> B[Feature Extraction <br> (e.g., Amplitude)]
    B --> C[Analysis Algorithm <br> (e.g., Threshold Detection)]
    C --> D[Output <br> (e.g., Sound Detected)]
```
- **System Overview**: The diagram shows an audio signal transformed into features, analyzed by an algorithm, and producing an output.
- **Component Relationships**: Feature extraction simplifies raw audio, the algorithm interprets features, and the output provides results.

## Implementation Details
### Basic Implementation
```python
import pyaudio
import numpy as np
import wave

# Audio capture parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3
OUTPUT_FILE = "recorded.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording for 3 seconds...")

# Capture audio
frames = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# Stop and close stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio to WAV file
wf = wave.open(OUTPUT_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Compute RMS energy for sound detection
samples = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
samples /= np.max(np.abs(samples))  # Normalize
rms = np.sqrt(np.mean(samples**2))

# Threshold-based detection
threshold = 0.1  # Adjust based on environment
if rms > threshold:
    print(f"Sound detected! RMS energy: {rms:.4f}")
else:
    print(f"No significant sound detected. RMS energy: {rms:.4f}")
```
- **Step-by-Step Setup**:
  1. Install Python (download from python.org).
  2. Install dependencies: `pip install pyaudio numpy`.
  3. Save the code as `audition_beginner.py`.
  4. Run the script: `python audition_beginner.py`.
- **Code Walkthrough**:
  - Uses `pyaudio` to capture 3 seconds of audio at 16 kHz, saves it as a WAV file, and computes RMS energy.
  - `pyaudio.open` configures the audio stream for mono recording.
  - RMS energy is calculated as the square root of the mean of squared samples.
  - A threshold-based rule detects loud sounds.
- **Common Pitfalls**:
  - Missing PyAudio (may require `portaudio` installation: `sudo apt-get install portaudio19-dev` on Linux, `brew install portaudio` on Mac).
  - No microphone connected or incorrect input device.
  - Audio buffer overflow if `CHUNK` size is incompatible with system.

## Real-World Applications
### Industry Examples
- **Use Case**: Clap detection in smart lights.
  - A device turns on/off when a loud clap is detected.
- **Implementation Patterns**: Capture audio, compute RMS energy, and use threshold detection to trigger actions.
- **Success Metrics**: High detection accuracy (>95%), low false positives in noisy environments.

### Hands-On Project
- **Project Goals**: Build a simple sound detector using Python.
- **Implementation Steps**:
  1. Run the above code to capture audio from your microphone.
  2. Make a loud sound (e.g., clap) during the 3-second recording.
  3. Check if the program detects the sound based on RMS energy.
  4. Adjust the threshold and test with quieter sounds to observe detection limits.
- **Validation Methods**: Confirm detection for loud sounds; verify saved WAV file plays correctly in an audio player.

## Tools & Resources
### Essential Tools
- **Development Environment**: Python, Jupyter notebooks for interactive coding.
- **Key Frameworks**: PyAudio for audio I/O, NumPy for numerical operations.
- **Testing Tools**: Audacity for audio file inspection, WAV file players for output verification.

### Learning Resources
- **Documentation**: PyAudio docs (https://people.csail.mit.edu/hubert/pyaudio/docs/), NumPy docs (https://numpy.org/doc/).
- **Tutorials**: Python audio processing basics (https://realpython.com/playing-and-recording-sound-python/).
- **Community Resources**: Reddit (r/learnpython), Stack Overflow for Python/PyAudio questions.

## References
- PyAudio documentation: https://people.csail.mit.edu/hubert/pyaudio/docs/
- NumPy documentation: https://numpy.org/doc/
- Audio signal processing basics: https://www.dsprelated.com/freebooks/sasp/
- Computer audition overview: https://en.wikipedia.org/wiki/Computational_audition

## Appendix
- **Glossary**:
  - **RMS Energy**: Root mean square of audio samples, indicating signal loudness.
  - **Sampling Rate**: Number of samples per second (e.g., 16 kHz).
  - **Feature Extraction**: Converting raw audio into numerical properties for analysis.
- **Setup Guides**:
  - Install PyAudio: `pip install pyaudio`.
  - Install NumPy: `pip install numpy`.
- **Code Templates**:
  - Frequency analysis: Use `numpy.fft.fft` for spectral features.
  - Real-time plotting: Use `matplotlib` to visualize audio waveforms.