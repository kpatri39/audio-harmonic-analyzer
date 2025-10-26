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
        Find the fundamental frequency with parabolic interpolation for sub-bin accuracy
        
        This method identifies the dominant frequency in the audio signal by finding the
        peak with maximum magnitude in the specified frequency range, then applies parabolic
        interpolation to achieve sub-bin frequency resolution.
        
        Algorithm:
        1. Filter frequency spectrum to valid range (min_freq to max_freq)
        2. Identify the bin with maximum magnitude (strongest peak)
        3. Apply three-point parabolic interpolation around the peak using magnitudes
        at bins k-1, k, and k+1 to estimate the true peak frequency between bins
        4. Return the interpolated frequency with sub-bin accuracy
        
        Parabolic Interpolation:
        Uses the formula δ = 0.5(α - γ)/(α - 2β + γ) where α, β, γ are magnitudes
        at bins k-1, k, k+1. The interpolated frequency is f[k] + δ·Δf, where Δf is
        the frequency bin spacing. This provides ~1-2 Hz accuracy even with coarse
        frequency resolution (e.g., 10.77 Hz bins with chunk_size=4096).
        
        Parameters:
        frequencies : numpy.ndarray
            Array of frequency bins in Hz from FFT output
        magnitudes : numpy.ndarray
            Array of magnitude values corresponding to each frequency bin
        min_freq : float, optional
            Minimum frequency to consider in Hz (default: 50)
            Filters out low-frequency noise below this threshold
        max_freq : float, optional
            Maximum frequency to consider in Hz (default: 2000)
            Limits search range for fundamental detection
        
        Returns:
        float
            Fundamental frequency in Hz with sub-bin accuracy from parabolic
            interpolation, or 0 if no valid frequencies found in the specified range
        """
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        
        valid_freqs = frequencies[mask]
        valid_mags = magnitudes[mask]

        if len(valid_mags) == 0:
            return 0
        
        # Find the STRONGEST peak (simplest approach)
        peak_idx = np.argmax(valid_mags)
        fundamental = valid_freqs[peak_idx]
        
        # Apply parabolic interpolation for sub-bin accuracy
        if peak_idx > 0 and peak_idx < len(valid_mags) - 1:
            # Get three points around the peak
            alpha = valid_mags[peak_idx - 1]
            beta = valid_mags[peak_idx]
            gamma = valid_mags[peak_idx + 1]
            
            # Calculate parabolic offset
            denominator = alpha - 2 * beta + gamma
            
            if abs(denominator) > 1e-10:  # Avoid division by zero
                delta = 0.5 * (alpha - gamma) / denominator
                
                # Clamp delta to prevent wild overshoots
                delta = np.clip(delta, -0.5, 0.5)
                
                # Get frequency bin spacing
                if len(frequencies) > 1:
                    freq_spacing = frequencies[1] - frequencies[0]
                    
                    # Calculate interpolated frequency
                    fundamental = fundamental + delta * freq_spacing
        
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

        octave = 4 + int(np.round(half_steps)) // 12
        note_idx = (int(np.round(half_steps)) + 9) % 12

        if note_idx < 0:
            note_idx += 12
            octave -= 1
        
        return f"{notes[note_idx]}{octave}"
    
    '''
    # COMMENTED OUT: Multi-note detection (experimental)
    # Challenges: 
    # - Harmonic ambiguity: cannot distinguish octaves (A4 2nd harmonic = A5 fundamental)
    # - Variable loudness: harmonics of loud note can overwhelm quiet note's fundamental
    # - Overlapping harmonics: different notes produce harmonics at similar frequencies
    # See theory section for mathematical limitations of polyphonic pitch detection
    
    def find_multiple_fundamentals(self, frequencies, magnitudes, min_freq=80, max_freq=2000, num_peaks=5):
        """
        Find multiple fundamental frequencies (for chords/double stops)
        
        Parameters:
        frequencies: frequency array from FFT
        magnitudes: magnitude array from FFT
        min_freq: minimum frequency to consider (default 80 Hz)
        max_freq: maximum frequency to consider (default 2000 Hz)
        num_peaks: maximum number of peaks to find (default 5)
        
        Returns:
        list of fundamental frequencies in Hz
        """
        # Filter to valid frequency range
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        valid_freqs = frequencies[mask]
        valid_mags = magnitudes[mask]
        
        if len(valid_mags) == 0:
            return []
        
        # Find peaks using scipy
        max_mag = np.max(valid_mags)
        peaks, properties = signal.find_peaks(
            valid_mags,
            prominence=max_mag * 0.15,
            height=max_mag * 0.3,
            distance=10
        )
        
        if len(peaks) == 0:
            return []
        
        # Get frequencies and magnitudes of peaks
        peak_freqs = valid_freqs[peaks]
        peak_mags = valid_mags[peaks]
        
        # Sort by magnitude (strongest first)
        sorted_indices = np.argsort(peak_mags)[::-1]
        sorted_peak_freqs = peak_freqs[sorted_indices][:num_peaks]
        sorted_peak_mags = peak_mags[sorted_indices][:num_peaks]
        
        # Filter out harmonics (improved version)
        fundamentals = []
        fundamental_mags = []
        
        for i in range(len(sorted_peak_freqs)):
            freq = sorted_peak_freqs[i]
            peak_mag = sorted_peak_mags[i]
            is_harmonic = False
            
            for j, fund in enumerate(fundamentals):
                fund_mag = fundamental_mags[j]
                ratio = freq / fund
                
                if abs(ratio - round(ratio)) < 0.15 and round(ratio) >= 2:
                    if peak_mag < fund_mag * 0.7:
                        is_harmonic = True
                        break
            
            if not is_harmonic:
                fundamentals.append(freq)
                fundamental_mags.append(peak_mag)
        
        return sorted(fundamentals)
    '''