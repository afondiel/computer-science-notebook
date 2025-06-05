# Computer Audition Technical Notes
A rectangular diagram depicting an intermediate computer audition pipeline, illustrating an audio input (e.g., environmental noise or speech) processed through advanced feature extraction (e.g., MFCCs, spectrograms), integrated into a machine learning pipeline with classification or detection algorithms (e.g., SVM or neural networks), trained with data augmentation, producing outputs like sound classification or event detection, annotated with preprocessing, model training, and evaluation metrics.

## Quick Reference
- **Definition**: Computer audition enables computers to analyze and interpret complex audio signals, such as classifying environmental sounds, detecting audio events, or performing basic speech recognition, using advanced signal processing and machine learning techniques implemented in C++.
- **Key Use Cases**: Real-time audio event detection, environmental sound classification, and voice command recognition in embedded systems.
- **Prerequisites**: Familiarity with C++ programming, basic signal processing (e.g., Fourier transforms), and introductory machine learning concepts (e.g., classification).

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: Computer audition involves programming computers to process audio signals for tasks like classifying sounds (e.g., siren vs. bird chirp) or detecting events in noisy environments using C++ for performance-critical applications.
- **Why**: It supports robust audio-based applications in embedded systems, IoT devices, and real-time analytics, leveraging C++ for efficiency.
- **Where**: Applied in smart home devices, security systems, automotive audio processing, and research for tasks like acoustic scene analysis.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - Audio signals are digitized time-series data, sampled at rates like 16 kHz, and processed to extract features for analysis.
  - Feature extraction transforms audio into representations like Mel-frequency cepstral coefficients (MFCCs) or spectrograms, capturing temporal and frequency patterns.
  - Machine learning models, such as Support Vector Machines (SVMs) or simple neural networks, are trained to classify or detect sounds, often with data augmentation to handle variability.
- **Key Components**:
  - **Preprocessing**: Noise reduction, normalization, or resampling to enhance audio quality.
  - **Feature Extraction**: Computing features like MFCCs, delta-MFCCs, or short-time Fourier transform (STFT) spectrograms.
  - **Data Augmentation**: Techniques like noise addition or time stretching to improve model robustness.
  - **Classification/Detection**: Algorithms to map features to labels or detect events, optimized for performance in C++.
- **Common Misconceptions**:
  - Misconception: Computer audition requires high-end hardware.
    - Reality: Efficient C++ implementations can run on resource-constrained devices with optimized libraries.
  - Misconception: Feature extraction is only for deep learning.
    - Reality: Features like MFCCs are effective for traditional machine learning models like SVMs.

### Visual Architecture
```mermaid
graph TD
    A[Audio Input <br> (e.g., Noise/Speech)] --> B[Preprocessing <br> (Noise Reduction, Resampling)]
    B --> C[Feature Extraction <br> (MFCC/Spectrogram)]
    C --> D[Pipeline <br> (SVM/Neural Network)]
    D -->|Evaluation| E[Output <br> (Classification/Detection)]
    F[Data Augmentation] --> B
    G[Performance Metrics] --> E
```
- **System Overview**: The diagram shows an audio signal preprocessed, transformed into features, fed into a machine learning pipeline, and producing classified or detected outputs.
- **Component Relationships**: Preprocessing and augmentation prepare audio, feature extraction enables modeling, and the pipeline delivers results.

## Implementation Details
### Intermediate Patterns
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <portaudio.h>
#include <fftw3.h>
#include <libsvm/svm.h>
#include <fstream>

// Error handling macro
#define PA_CHECK(err) if(err != paNoError) { std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl; exit(1); }

// Callback for audio capture
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo* timeInfo,
                        PaStreamCallbackFlags statusFlags,
                        void* userData) {
    const float* input = (const float*)inputBuffer;
    std::vector<float>* samples = (std::vector<float>*)userData;
    samples->insert(samples->end(), input, input + framesPerBuffer);
    return paContinue;
}

// Compute simple MFCC-like features (approximation using FFT)
std::vector<double> extractFeatures(const std::vector<float>& samples, int sr) {
    const int n_fft = 512;
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_plan plan = fftw_plan_dft_1d(n_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    std::vector<double> features;
    for(size_t i = 0; i < samples.size(); i += n_fft/2) {
        // Windowed FFT
        for(int j = 0; j < n_fft && i + j < samples.size(); ++j) {
            double window = 0.5 * (1 - std::cos(2 * M_PI * j / (n_fft - 1))); // Hann window
            in[j][0] = samples[i + j] * window;
            in[j][1] = 0.0;
        }
        fftw_execute(plan);

        // Compute log-magnitude spectrum (simplified MFCC)
        std::vector<double> mag(n_fft/2);
        for(int j = 0; j < n_fft/2; ++j) {
            mag[j] = std::log(std::sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) + 1e-10);
        }
        // Average across bands (simulating Mel filterbank)
        double avg = 0.0;
        for(double m : mag) avg += m;
        features.push_back(avg / mag.size());
        if(features.size() >= 13) break; // Limit to 13 features
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return features;
}

int main() {
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    PA_CHECK(err);

    // Configure stream
    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // Open stream
    PaStream* stream;
    std::vector<float> samples;
    err = Pa_OpenStream(&stream, &inputParams, nullptr, 16000, 256, paNoFlag, audioCallback, &samples);
    PA_CHECK(err);

    // Record for 3 seconds
    err = Pa_StartStream(stream);
    PA_CHECK(err);
    std::cout << "Recording for 3 seconds..." << std::endl;
    Pa_Sleep(3000);
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    // Extract features
    std::vector<double> features = extractFeatures(samples, 16000);

    // Simulate SVM classification (trained model placeholder)
    svm_model* model = svm_load_model("svm_model.txt"); // Assume pre-trained model
    if(!model) {
        std::cerr << "Failed to load SVM model" << std::endl;
        return 1;
    }

    // Prepare SVM input
    std::vector<svm_node> svm_input(features.size() + 1);
    for(size_t i = 0; i < features.size(); ++i) {
        svm_input[i].index = i + 1;
        svm_input[i].value = features[i];
    }
    svm_input[features.size()].index = -1; // End of features

    // Predict
    double prediction = svm_predict(model, svm_input.data());
    std::cout << "Predicted class: " << (prediction == 1 ? "Sound Event" : "No Event") << std::endl;

    svm_free_and_destroy_model(&model);

    // Save audio for verification
    std::ofstream out("recorded.wav", std::ios::binary);
    int chunk_size = 36 + samples.size() * 4;
    out.write("RIFF", 4);
    out.write((char*)&chunk_size, 4);
    out.write("WAVEfmt ", 8);
    int fmt_size = 16;
    out.write((char*)&fmt_size, 4);
    short audio_format = 3;
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
  2. Install PortAudio: `sudo apt-get install libportaudio2 libportaudio-dev` (Linux), `brew install portaudio` (Mac), or download from portaudio.com (Windows).
  3. Install FFTW: `sudo apt-get install libfftw3-dev` (Linux), `brew install fftw` (Mac), or download from fftw.org (Windows).
  4. Install libSVM: Download from https://www.csie.ntu.edu.tw/~cjlin/libsvm/, build, and link.
  5. Save code as `audition_intermediate.cpp`.
  6. Compile: `g++ -o audition audition_intermediate.cpp -lportaudio -lfftw3 -lsvm`.
  7. Run: `./audition` (Linux/Mac) or `audition.exe` (Windows).
- **Code Walkthrough**:
  - Uses PortAudio to capture 3 seconds of audio at 16 kHz.
  - Computes simplified MFCC-like features using FFTW for spectral analysis with a Hann window.
  - Applies a pre-trained SVM model (placeholder) to classify audio as "Sound Event" or "No Event".
  - Saves audio as a WAV file for verification.
- **Common Pitfalls**:
  - Missing FFTW or libSVM libraries during compilation or linking.
  - Incorrect microphone setup or sample rate mismatch.
  - SVM model file (`svm_model.txt`) must exist and match feature dimensions.

## Real-World Applications
### Industry Examples
- **Use Case**: Environmental sound classification in smart cities.
  - Detects sounds like sirens or horns for traffic management.
- **Implementation Patterns**: Extract MFCCs, train an SVM, and deploy on embedded devices using C++.
- **Success Metrics**: >90% classification accuracy, <100ms latency.

### Hands-On Project
- **Project Goals**: Build a sound classifier for two audio classes (e.g., clap vs. whistle).
- **Implementation Steps**:
  1. Record 10 clap and 10 whistle samples (WAV, ~2 seconds, 16 kHz) using Audacity.
  2. Use the above code to extract features from each sample.
  3. Train an SVM model using libSVM tools (e.g., `svm-train`) on extracted features.
  4. Test the code with new recordings and verify classification output.
- **Validation Methods**: Achieve >85% accuracy; confirm WAV output is audible.

## Tools & Resources
### Essential Tools
- **Development Environment**: C++ compiler (g++, clang), IDE (Visual Studio Code).
- **Key Frameworks**: PortAudio for audio I/O, FFTW for Fourier transforms, libSVM for machine learning.
- **Testing Tools**: Audacity for audio inspection, WAV players for output verification.

### Learning Resources
- **Documentation**: PortAudio (http://www.portaudio.com/docs.html), FFTW (http://www.fftw.org/doc/), libSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/).
- **Tutorials**: C++ signal processing (https://www.dsprelated.com/freebooks/sasp/).
- **Community Resources**: r/cpp, Stack Overflow for C++/PortAudio/FFTW questions.

## References
- PortAudio documentation: http://www.portaudio.com/docs.html
- FFTW documentation: http://www.fftw.org/doc/
- libSVM documentation: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- Computer audition overview: https://en.wikipedia.org/wiki/Computational_audition

## Appendix
- **Glossary**:
  - **MFCC**: Mel-frequency cepstral coefficients, features for audio analysis.
  - **STFT**: Short-time Fourier transform, basis for spectrograms.
  - **Data Augmentation**: Modifying audio to enhance model robustness.
- **Setup Guides**:
  - Install FFTW on Linux: `sudo apt-get install libfftw3-dev`.
  - Install libSVM: Follow instructions at https://www.csie.ntu.edu.tw/~cjlin/libsvm/.
- **Code Templates**:
  - Spectrogram computation: Use FFTW to generate time-frequency plots.
  - Neural network: Implement a simple MLP with custom C++ code.