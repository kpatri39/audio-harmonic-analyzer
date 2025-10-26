# pylint: disable=all
# type: ignore

import pyaudio
import numpy as np
from fft_engine import FFTAnalyzer

class AudioCapture:
    """
    Captures real-time audio from microphone and analyzes it with FFT
    """
    
    def __init__(self, sample_rate=44100, chunk_size=4096, device_index=None):
        """
        Initialize audio capture
        
        Parameters:
        sample_rate: Audio sampling rate in Hz (default 44100)
        chunk_size: Number of samples per audio chunk (default 4096)
        device_index: Specific device index to use (None = default device)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.analyzer = FFTAnalyzer(sample_rate=sample_rate)
        
        self.audio = pyaudio.PyAudio()
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size
        )
        
        device_name = "default"
        if device_index is not None:
            device_info = self.audio.get_device_info_by_index(device_index)
            device_name = device_info['name']
        
        print(f"Audio capture initialized: {sample_rate} Hz, chunk size {chunk_size}")
        print(f"Using device: {device_name}")
    
    def list_audio_devices(self):
        """
        List all available audio input devices
        
        Returns:
        List of (device_index, device_name) tuples
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append((i, device_info['name']))
                print(f"Device {i}: {device_info['name']}")
        return devices
    
    def read_chunk(self):
        """
        Read one chunk of audio from microphone
        
        Returns:
        numpy array of audio samples (normalized to range -1.0 to 1.0)
        """
        raw_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

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
        - audio_data: raw audio samples
        """
        audio_data = self.read_chunk()
        
        frequencies, magnitudes = self.analyzer.compute_fft(audio_data)
        fundamental = self.analyzer.find_fundamental_frequency(frequencies, magnitudes)
        note = self.analyzer.frequency_to_note(fundamental)
        harmonics = self.analyzer.find_harmonics(frequencies, magnitudes, fundamental)
        
        return {
            'fundamental': fundamental,
            'note': note,
            'harmonics': harmonics,
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'audio_data': audio_data
        }
    
    def close(self):
        """
        Clean up audio resources
        """
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("Audio capture closed")