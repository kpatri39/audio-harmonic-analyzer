# pylint: disable=all
# type: ignore

import pyaudio
import numpy as np
from fft_engine import FFTAnalyzer

class AudioCapture:
    """
    Captures real-time audio from microphone and analyzes it with FFT
    """

    def __init__(self, sample_rate=44100, chunk_size=4096):
        """
        Initialize audio capture
        
        Parameters:
        sample_rate: Audio sampling rate in Hz (default 44100)
        chunk_size: Number of samples per audio chunk (default 4096)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.analyzer = FFTAnalyzer(sample_rate=sample_rate)

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)

        print(f"Audio capture initialized: {sample_rate} Hz, chunk size {chunk_size}")
    
    def read_chunk(self):
        """
        Read one chunk of audio from microphone
        
        Returns:
        numpy array of audio samples (normalized to range -1.0 to 1.0)
        """
        # Read raw bytes from stream
        raw_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        
        # Convert bytes to numpy array of 16-bit integers
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        
        # Normalize to range -1.0 to 1.0
        audio_data = audio_data.astype(np.float32) / 2**15
        
        return audio_data
    
    def analyze_chunk(self):
        """
        Read and analyze one chunk of audio
        
        Returns:
        Dictionary with analysis results:
        - fundamental: fundamental frequency in Hz
        - note: note name
        - harmonics: list of (harmonic_number, frequency, magnitude)
        - frequencies: frequency array from FFT
        - magnitudes: magnitude array from FFT
        """
        audio_data = self.read_chunk()

        frequencies, magnitudes = self.analyzer.compute_fft(audio_data)
        fundamental = self.analyzer.find_fundamental_frequency(frequencies, magnitudes)
        note = self.analyzer.frequency_to_note(fundamental)
        harmonics = self.analyzer.find_harmonics(frequencies, magnitudes, fundamental)

        return {'fundamental': fundamental, 'note': note, 'harmonics': harmonics, 'frequencies': frequencies, 'magnitudes': magnitudes, 'audio_data': audio_data}
    
    def close(self):
        """
        Clean up audio resources
        """
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("Audio capture closed")