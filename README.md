# Real-Time FFT Audio Analyzer

A real-time frequency analysis tool that uses Fast Fourier Transform (FFT) to identify musical notes and visualize their harmonic structure. Built for analyzing violin acoustics and understanding the mathematical foundations of audio signal processing.

## Overview

This project implements a complete audio analysis pipeline that captures live audio, performs FFT analysis, and displays real-time visualizations of both time-domain waveforms and frequency-domain spectra. The analyzer uses parabolic interpolation to achieve sub-bin frequency accuracy, enabling precise pitch detection even with modest computational resources.

## Features

- **Real-time audio capture** from system microphone or external audio interface
- **Fast Fourier Transform analysis** with Hann windowing to minimize spectral leakage
- **Sub-bin frequency accuracy** using parabolic interpolation (~1-2 Hz precision)
- **Musical note identification** with automatic octave detection
- **Harmonic series visualization** showing overtone structure
- **Dual visualization**: 
  - Time-domain waveform display
  - Frequency spectrum with magnitude filtering
- **Optimized rendering** using matplotlib blitting for smooth real-time updates
- **Modern dark-themed GUI** built with Tkinter

## Technical Highlights

- **Parabolic interpolation** for frequency precision beyond FFT bin resolution
- **Adaptive magnitude thresholding** to filter background noise
- **Threaded audio processing** to maintain responsive GUI
- **Support for external microphones** with device selection capability

## Requirements
```
Python 3.7+
numpy
scipy
matplotlib
pyaudio
tkinter (usually included with Python)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kpatri39/audio-harmonic-analyzer.git
cd audio-harmonic-analyzer
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib pyaudio
```

## Usage

Run the main GUI application:
```bash
python3 gui.py
```

1. Click the **▶ Start** button to begin audio capture
2. Play your instrument or sing into the microphone
3. Observe the detected note, frequency, waveform, and spectrum in real-time
4. Click **⏸ Stop** to pause analysis

### Testing Individual Components

Test the FFT engine with synthetic signals:
```bash
python3 test_fft.py
```

Test audio capture in terminal:
```bash
python3 test_live_audio.py
```

## Project Structure
```
audio_harmonic_analyzer/
├── fft_engine.py          # Core FFT analysis and pitch detection algorithms
├── audio_capture.py       # Real-time audio input and processing
├── gui.py                 # Tkinter GUI with matplotlib visualization
├── test_fft.py           # Unit tests for FFT engine with synthetic signals
├── test_live_audio.py    # Terminal-based audio testing
└── THEORY.md             # Comprehensive mathematical documentation
```

## How It Works

The analyzer operates in three stages:

1. **Audio Capture**: PyAudio streams audio from the microphone at 44.1 kHz, capturing chunks of 4096 samples (~93ms windows)

2. **FFT Analysis**: 
   - Applies Hann window to reduce spectral leakage
   - Computes Fast Fourier Transform to convert time → frequency domain
   - Extracts magnitude spectrum showing frequency content

3. **Pitch Detection**:
   - Identifies strongest peak in valid frequency range (50-2000 Hz)
   - Applies 3-point parabolic interpolation for sub-bin accuracy
   - Converts frequency to musical note using equal temperament formula

## Mathematical Foundation

The core of this project relies on the **Discrete Fourier Transform** (DFT), which decomposes a time-domain signal into its constituent frequencies. The Fast Fourier Transform (FFT) efficiently computes the DFT in O(N log N) time.

Musical note conversion uses the **equal temperament** system, where each semitone is related by a frequency ratio of the 12th root of 2. The number of semitones between a frequency and A4 (440 Hz) is calculated using:

**semitones = 12 × log₂(frequency / 440)**

**Parabolic interpolation** refines the peak frequency estimate by fitting a parabola through three points around the detected peak, achieving sub-bin accuracy of ~1-2 Hz even with coarse frequency resolution.

For complete mathematical derivations with equations, algorithm explanations, and discussion of limitations, see [THEORY.md](THEORY.md).

## Limitations & Known Issues

- **Low-frequency detection**: For low-pitched instruments (e.g., violin notes below C3), harmonics may be stronger than the fundamental, leading to octave errors
- **Polyphonic content**: Currently designed for monophonic pitch detection; chord/multi-note detection is not reliable (see commented implementation in code)
- **Harmonic ambiguity**: Cannot distinguish between octaves when the 2nd harmonic matches another note's fundamental
- **Instrument-specific behavior**: Bright instruments (e.g., violin E strings) may have harmonics that compete with the fundamental in magnitude

These limitations are fundamental to FFT-based pitch detection and represent active research areas in Music Information Retrieval. See THEORY.md for detailed discussion.

## Real-World Applications

This analyzer reveals the acoustic signature of musical instruments:
- Visualize how bow pressure and position affect harmonic content
- Compare frequency responses of different microphones
- Understand why certain instruments sound "bright" vs "mellow"
- Analyze the effect of string quality on harmonic structure

## Future Enhancements

- [ ] Pitch tracking over time with visual history
- [ ] Tuner mode with cent deviation display
- [ ] Recording and export of analysis data
- [ ] Improved polyphonic pitch detection using advanced algorithms
- [ ] Spectrogram view for time-frequency analysis
- [ ] Formant analysis for voice characterization

## Author

Kartik Patri - Northeastern University

## Acknowledgments

- Inspired by professional audio analysis tools and music information retrieval research
- Built as an exploration of applied Fourier analysis and real-time signal processing

## License

MIT License - feel free to use and modify for educational purposes
