# pylint: disable=all
# type: ignore

import numpy as np
from fft_engine import FFTAnalyzer

# Generate a test tone (A4 = 440 Hz)
sample_rate = 44100
duration = 1.0  # seconds
frequency = 440.0  # A4 note

t = np.linspace(0, duration, int(sample_rate * duration))

audio_data = np.sin(2 * np.pi * frequency * t)

analyzer = FFTAnalyzer(sample_rate=sample_rate)

frequencies, magnitudes = analyzer.compute_fft(audio_data)

fundamental = analyzer.find_fundamental_frequency(frequencies, magnitudes)

note = analyzer.frequency_to_note(fundamental)

print(f"Generated frequency: {frequency} Hz")
print(f"Detected frequency: {fundamental:.2f} Hz")
print(f"Detected note: {note}")

harmonics = analyzer.find_harmonics(frequencies, magnitudes, fundamental)
print(f"\nHarmonics:")
for n, freq, mag in harmonics:
    print(f"  Harmonic {n}: {freq:.2f} Hz (magnitude: {mag:.2f})")