# Computer Audition Technical Notes
A rectangular diagram illustrating the computer audition process, showing an audio input (e.g., a sound wave from a microphone) processed through basic feature extraction (e.g., amplitude analysis), analyzed with a simple algorithm (e.g., threshold-based detection), producing an output (e.g., identifying a loud sound), with arrows indicating the flow from audio capture to processing to result.

## Quick Reference
- **Definition**: Computer audition is the field of enabling computers to analyze and interpret audio signals, such as detecting sounds, recognizing speech, or classifying noises, using computational techniques.
- **Key Use Cases**: Sound detection in smart devices, basic speech recognition, and environmental noise monitoring.
- **Prerequisites**: Basic understanding of C++ programming, familiarity with audio as sound waves, and introductory knowledge of signal processing.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: Computer audition involves programming computers to "listen" to audio inputs and perform tasks like detecting a clap, identifying a siren, or transcribing simple speech.
- **Why**: It enables applications like voice-controlled devices, audio-based alerts, and automated sound analysis in various environments.
- **Where**: Used in IoT devices, security systems, mobile apps, and research for tasks like audio event detection or basic voice interaction.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - Audio is captured as a digital signal, a sequence of numbers representing sound wave amplitude over time, typically sampled at rates like 16 kHz.
  - Computer audition processes these signals by extracting simple features (e.g., amplitude or energy) and applying algorithms to interpret them.
  - Basic algorithms, like threshold-based detection, can identify sound events by comparing features to predefined values.
- **Key Components**:
  - **Audio Capture**: Recording sound using a microphone, converted to digital samples via an Analog-to-Digital Converter (ADC).
  - **Feature Extraction**: Computing basic properties like amplitude or root mean square (RMS) energy from audio samples.
  - **Analysis Algorithm**: A simple decision rule (e.g., if energy exceeds a threshold, detect a sound) to produce meaningful outputs.
- **Common Misconceptions**:
  - Misconception: Computer audition is only about speech recognition.
    - Reality: It includes non-speech sounds like environmental noises or musical notes.
  - Misconception: You need advanced hardware to start.
    - Reality: Basic audition tasks can be done with standard computers and microphones using C++ libraries.

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
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <portaudio.h>
#include <fstream>

// Error handling macro
#define PA_CHECK(err) if(err != paNoError) { std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl; exit(1); }

// Callback function for audio capture
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo* timeInfo,
                        PaStreamCallbackFlags statusFlags,
                        void* userData) {
    const float* input = (const float*)inputBuffer;
    std::vector<float>* samples = (std::vector<float>*)userData;

    // Store samples
    for(unsigned long i = 0; i < framesPerBuffer; i++) {
        samples->push_back(input[i]);
    }
    return paContinue;
}

int main() {
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    PA_CHECK(err);

    // Configure stream parameters
    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    inputParams.channelCount = 1; // Mono
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // Open stream
    PaStream* stream;
    std::vector<float> samples;
    err = Pa_OpenStream(&stream, &inputParams, nullptr, 16000, 256, paNoFlag, audioCallback, &samples);
    PA_CHECK(err);

    // Start recording
    err = Pa_StartStream(stream);
    PA_CHECK(err);
    std::cout << "Recording for 3 seconds..." << std::endl;
    Pa_Sleep(3000); // Record for 3 seconds

    // Stop and close stream
    err = Pa_StopStream(stream);
    PA_CHECK(err);
    err = Pa_CloseStream(stream);
    PA_CHECK(err);
    Pa_Terminate();

    // Compute RMS energy for sound detection
    float sum_squares = 0.0f;
    for(float sample : samples) {
        sum_squares += sample * sample;
    }
    float rms = std::sqrt(sum_squares / samples.size());

    // Threshold-based detection
    float threshold = 0.05f; // Adjust based on environment
    if(rms > threshold) {
        std::cout << "Sound detected! RMS energy: " << rms << std::endl;
    } else {
        std::cout << "No significant sound detected. RMS energy: " << rms << std::endl;
    }

    // Save samples to file (optional)
    std::ofstream out("recorded.wav", std::ios::binary);
    // Simplified WAV header (mono, 16 kHz, 32-bit float)
    out.write("RIFF", 4);
    int chunk_size = 36 + samples.size() * 4;
    out.write((char*)&chunk_size, 4);
    out.write("WAVEfmt ", 8);
    int fmt_size = 16;
    out.write((char*)&fmt_size, 4);
    short audio_format = 3; // IEEE float
    out.write((char*)&audio_format, 2);
    short channels = 1;
    out.write((char*)&channels, 2);
    int sample_rate = 16000;
    out.write((char*)&sample_rate, 4);
    int byte_rate = 16000 * 1 * 4;
    out.write((char*)&byte_rate, 4);
    short block_align = 4;
    out.write((char*)&block_align, 2);
    short bits_per_sample = 32;
    out.write((char*)&bits_per_sample, 2);
    out.write("data", 4);
    int data_size = samples.size() * 4;
    out.write((char*)&data_size, 4);
    out.write((char*)samples.data(), samples.size() * 4);
    out.close();

    return 0;
}
```
- **Step-by-Step Setup**:
  1. Install a C++ compiler (e.g., g++ on Linux/Mac, MSVC on Windows).
  2. Install PortAudio: On Linux, `sudo apt-get install libportaudio2 libportaudio-dev`; on Mac, `brew install portaudio`; on Windows, download from portaudio.com.
  3. Save the code as `audition_beginner.cpp`.
  4. Compile: `g++ -o audition audition_beginner.cpp -lportaudio`.
  5. Run: `./audition` (Linux/Mac) or `audition.exe` (Windows).
- **Code Walkthrough**:
  - The code uses PortAudio to capture 3 seconds of audio at 16 kHz, computes RMS energy, and detects sound if energy exceeds a threshold.
  - `audioCallback` collects audio samples from the microphone.
  - RMS energy is calculated as the square root of the mean of squared samples.
  - A simple WAV file is saved with a basic header for playback verification.
- **Common Pitfalls**:
  - Missing PortAudio library or incorrect linking during compilation.
  - No microphone connected or incorrect default input device.
  - Incorrect WAV header causing playback issues in audio players.

## Real-World Applications
### Industry Examples
- **Use Case**: Clap detection in smart lights.
  - A device turns on/off when a loud clap is detected.
- **Implementation Patterns**: Capture audio, compute RMS energy, and use threshold detection to trigger actions.
- **Success Metrics**: High detection accuracy (>95%), low false positives in noisy environments.

### Hands-On Project
- **Project Goals**: Build a simple sound detector using C++.
- **Implementation Steps**:
  1. Set up the above code to capture audio from your microphone.
  2. Run the program and make a loud sound (e.g., clap) during the 3-second recording.
  3. Check if the program detects the sound based on RMS energy.
  4. Adjust the threshold and test with quieter sounds to observe detection limits.
- **Validation Methods**: Confirm detection for loud sounds; verify saved WAV file plays correctly in an audio player.

## Tools & Resources
### Essential Tools
- **Development Environment**: C++ compiler (e.g., g++, clang), IDE (e.g., Visual Studio Code).
- **Key Frameworks**: PortAudio for cross-platform audio I/O.
- **Testing Tools**: Audacity for audio file inspection, WAV file players for output verification.

### Learning Resources
- **Documentation**: PortAudio docs (http://www.portaudio.com/docs.html).
- **Tutorials**: C++ audio programming basics (https://www.dsprelated.com/freebooks/sasp/).
- **Community Resources**: Stack Overflow for C++/PortAudio questions, r/cpp for programming support.

## References
- PortAudio documentation: http://www.portaudio.com/docs.html
- Audio signal processing basics: https://www.dsprelated.com/freebooks/sasp/
- WAV file format: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
- Computer audition overview: https://en.wikipedia.org/wiki/Computational_audition

## Appendix
- **Glossary**:
  - **RMS Energy**: Root mean square of audio samples, indicating signal loudness.
  - **Sampling Rate**: Number of samples per second (e.g., 16 kHz).
  - **Feature Extraction**: Converting raw audio into numerical properties for analysis.
- **Setup Guides**:
  - Install PortAudio on Linux: `sudo apt-get install libportaudio2 libportaudio-dev`.
  - Install g++: `sudo apt-get install g++` (Linux) or download Xcode (Mac).
- **Code Templates**:
  - Frequency analysis: Use FFT libraries like FFTW for spectral features.
  - Stereo capture: Modify `channelCount` to 2 in PortAudio stream parameters.