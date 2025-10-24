# pylint: disable=all
# type: ignore

import numpy as np
from scipy import signal

class FFTAnalyzer:
    """
    Analyzes audio signals using Fast Fourier Transform to identify
    frequencies, harmonics, and musical notes.
    """

    def __init__(self, sample_rate=44100):
        """
        Initialize the FFT analyzer
        
        Parameters:
        sample_rate: Audio sampling rate in Hz (default 44100, CD quality)
        """
        self.sample_rate = sample_rate
    
    def compute_fft(self, audio_data):
        """
        Compute FFT of audio data

        Steps:
        1. Apply Hann window to audio data to reduce spectral leakage
        2. Compute FFT to transform time-domain signal to frequency domain
        3. Extract magnitude values from complex FFT output
        4. Calculate corresponding frequency bins

        Parameters:
        audio_data: numpy array of audio samples

        Returns:
        frequencies: array of frequency bins in Hz
        magnitudes: array of magnitude values for each frequency
        """
        windowed_data = audio_data * np.hanning(len(audio_data))
        fft_result = np.fft.rfft(windowed_data)
        
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(len(audio_data), 1.0/self.sample_rate)

        return frequencies, magnitudes
    
    def find_fundamental_frequency(self, frequencies, magnitudes, min_freq=50, max_freq=2000):
        """
        Find the fundamental frequency (strongest peak in spectrum)
        
        Steps:
        1. Filter frequencies to valid range (min_freq to max_freq)
        2. Extract corresponding magnitudes for valid frequencies
        3. Find the frequency with maximum magnitude
        4. Return that frequency as the fundamental
        
        Parameters:
        frequencies: frequency array from FFT
        magnitudes: magnitude array from FFT
        min_freq: minimum frequency to consider in Hz (default 50)
        max_freq: maximum frequency to consider in Hz (default 2000)
        
        Returns:
        fundamental frequency in Hz, or 0 if no valid frequencies found
        """
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        
        valid_freqs = frequencies[mask]
        valid_mags = magnitudes[mask]

        if len(valid_mags) == 0:
            return 0
        
        peak_idx = np.argmax(valid_mags)
        fundamental = valid_freqs[peak_idx]

        return fundamental
    
    def find_harmonics(self, frequencies, magnitudes, fundamental, num_harmonics=5):
        """
        Find harmonic series based on fundamental frequency
        
        Steps:
        1. For each harmonic number (1, 2, 3, 4, 5...)
        2. Calculate expected frequency (fundamental × harmonic_number)
        3. Find the closest actual frequency in FFT output
        4. Store the harmonic number, frequency, and magnitude
        
        Parameters:
        frequencies: frequency array from FFT
        magnitudes: magnitude array from FFT  
        fundamental: fundamental frequency in Hz
        num_harmonics: number of harmonics to find (default 5)
        
        Returns:
        list of (harmonic_number, frequency, magnitude) tuples
        """
        harmonics = []

        for i in range(1, num_harmonics + 1):
            expected_freq = fundamental * i
            
            idx = np.argmin(np.abs(frequencies - expected_freq))
            actual_freq = frequencies[idx]
            magnitude = magnitudes[idx]

            harmonics.append((i, actual_freq, magnitude))
        
        return harmonics
    
    def frequency_to_note(self, frequency):
        """
        Convert frequency to musical note name
        
        Steps:
        1. Calculate semitones away from A4 (440 Hz) using logarithm
        2. Determine octave number from semitone offset
        3. Determine note position within octave
        4. Return note name with octave
        
        Parameters:
        frequency: frequency in Hz
        
        Returns:
        note name as string (e.g., "A4", "C#5", "G3")
        """
        if frequency <= 0:
            return "N/A"
        
        A4 = 440.0
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        half_steps = 12 * np.log2(frequency / A4)

        octave = 4 + int(np.round(half_steps) // 12)
        note_idx = (int(np.round(half_steps)) + 9) % 12

        if note_idx < 0:
            note_idx += 12
            octave -= 1
        
        return f"{notes[note_idx]}{octave}"