# pylint: disable=all
# type: ignore

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from audio_capture import AudioCapture
import threading

class FFTAnalyzerGUI:
    """
    Real-time FFT analyzer GUI with spectrum visualization
    """
    
    def __init__(self, root):
        """
        Initialize the GUI
        
        Parameters:
        root: tkinter root window
        """
        self.root = root
        self.root.title("FFT Audio Analyzer")
        self.root.geometry("1000x850")
        self.root.configure(bg='#1e1e1e')
        
        self.capture = AudioCapture()
        self.running = False
        
        self.background = None
        self.background_wave = None
        self.background_spectrum = None
        
        self.setup_style()
    
    def setup_style(self):
        """
        Configure modern styling
        """
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TButton', 
                       font=('Arial', 12, 'bold'),
                       padding=10)
    
    def create_widgets(self):
        """
        Create all GUI widgets
        """
        control_frame = tk.Frame(self.root, bg='#1e1e1e', pady=15)
        control_frame.grid(row=0, column=0)
        
        self.start_button = ttk.Button(control_frame, text="▶ Start", command=self.toggle_analysis)
        self.start_button.pack()

        info_frame = tk.Frame(self.root, bg='#2d2d2d', pady=20, padx=20)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=20, pady=10)
        
        self.note_label = tk.Label(info_frame, text="--", 
                                   font=("Arial", 80, "bold"), 
                                   fg="#00d4ff",
                                   bg='#2d2d2d')
        self.note_label.pack()
        
        self.freq_label = tk.Label(info_frame, text="--- Hz", 
                                   font=("Arial", 26),
                                   fg="#ffffff",
                                   bg='#2d2d2d')
        self.freq_label.pack(pady=(5, 0))
        
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor='#1e1e1e')
        
        self.ax_wave = self.fig.add_subplot(211, facecolor='#2d2d2d')
        self.ax_wave.set_xlabel('Time (samples)', color='#ffffff', fontsize=10)
        self.ax_wave.set_ylabel('Amplitude', color='#ffffff', fontsize=10)
        self.ax_wave.set_title('Waveform', color='#00d4ff', fontsize=12, fontweight='bold')
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.grid(True, alpha=0.3, color='#444444')
        self.ax_wave.tick_params(colors='#ffffff', labelsize=8)
        self.waveform_line, = self.ax_wave.plot([], [], '#00ff88', linewidth=1, animated=True)
        
        self.ax_spectrum = self.fig.add_subplot(212, facecolor='#2d2d2d')
        self.ax_spectrum.set_xlabel('Frequency (Hz)', color='#ffffff', fontsize=10)
        self.ax_spectrum.set_ylabel('Magnitude', color='#ffffff', fontsize=10)
        self.ax_spectrum.set_title('Frequency Spectrum', color='#00d4ff', fontsize=12, fontweight='bold')
        self.ax_spectrum.set_xlim(0, 2000)
        self.ax_spectrum.set_ylim(0, 1000)
        self.ax_spectrum.grid(True, alpha=0.3, color='#444444')
        self.ax_spectrum.tick_params(colors='#ffffff', labelsize=8)
        self.spectrum_line, = self.ax_spectrum.plot([], [], '#ff4444', linewidth=2, animated=True)
        
        self.fig.tight_layout(pad=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg='#1e1e1e')
        canvas_widget.grid(row=2, column=0, padx=20, pady=10)
        
        self.canvas.draw()
        self.background_wave = self.canvas.copy_from_bbox(self.ax_wave.bbox)
        self.background_spectrum = self.canvas.copy_from_bbox(self.ax_spectrum.bbox)
    
    def toggle_analysis(self):
        """
        Start or stop the audio analysis
        """
        if not self.running:
            self.running = True
            self.start_button.config(text="⏸ Stop")
            
            self.thread = threading.Thread(target=self.process_audio, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self.start_button.config(text="▶ Start")
    
    def process_audio(self):
        """
        Main audio processing loop (runs in separate thread)
        """
        while self.running:
            results = self.capture.analyze_chunk()
            
            self.root.after(0, self.update_display, results)
    
    def update_display(self, results):
        """
        Update GUI with analysis results using blitting for speed
        
        Parameters:
        results: dictionary from analyze_chunk()
        """
        fundamental = results['fundamental']
        note = results['note']
        frequencies = results['frequencies']
        magnitudes = results['magnitudes']
        audio_data = results['audio_data']
        
        if fundamental > 0:
            self.note_label.config(text=note)
            self.freq_label.config(text=f"{fundamental:.2f} Hz")
        else:
            self.note_label.config(text="--")
            self.freq_label.config(text="--- Hz")
        
        time_axis = np.arange(len(audio_data))
        self.waveform_line.set_data(time_axis, audio_data)
        self.ax_wave.set_xlim(0, len(audio_data))
        
        mask = magnitudes > 10
        filtered_freqs = frequencies[mask]
        filtered_mags = magnitudes[mask]
        
        self.spectrum_line.set_data(filtered_freqs, filtered_mags)
        
        if len(filtered_mags) > 0:
            max_mag = np.max(filtered_mags)
            current_ylim = self.ax_spectrum.get_ylim()[1]
            if max_mag > current_ylim * 0.9 or max_mag < current_ylim * 0.3:
                self.ax_spectrum.set_ylim(0, max_mag * 1.1)
                self.canvas.draw()
                self.background_wave = self.canvas.copy_from_bbox(self.ax_wave.bbox)
                self.background_spectrum = self.canvas.copy_from_bbox(self.ax_spectrum.bbox)
                return
        
        self.canvas.restore_region(self.background_wave)
        self.canvas.restore_region(self.background_spectrum)
        
        self.ax_wave.draw_artist(self.waveform_line)
        self.ax_spectrum.draw_artist(self.spectrum_line)
        
        self.canvas.blit(self.ax_wave.bbox)
        self.canvas.blit(self.ax_spectrum.bbox)
        
        self.canvas.flush_events()
    
    def close(self):
        """
        Clean up when closing the application
        """
        self.running = False
        self.capture.close()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FFTAnalyzerGUI(root)
    app.create_widgets()
    
    root.protocol("WM_DELETE_WINDOW", app.close)
    
    root.mainloop()