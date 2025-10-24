# pylint: disable=all
# type: ignore

import time
import numpy as np
from audio_capture import AudioCapture

# Initialize audio capture
capture = AudioCapture()

print("\nPlay your violin! Press Ctrl+C to stop.\n")

try:
    while True:
        # Analyze current audio chunk
        results = capture.analyze_chunk()
        
        # Print results only if fundamental is valid AND magnitude is strong enough
        if results['fundamental'] > 0:
            # Find the magnitude of the fundamental frequency
            fund_freq = results['fundamental']
            freqs = results['frequencies']
            mags = results['magnitudes']
            
            # Get magnitude at fundamental
            idx = np.argmin(np.abs(freqs - fund_freq))
            fundamental_magnitude = mags[idx]
            
            # Only print if magnitude is above threshold (adjust this value as needed)
            if fundamental_magnitude > 9:  # Experiment with this threshold
                print(f"Note: {results['note']:4s} | Frequency: {results['fundamental']:7.2f} Hz | Mag: {fundamental_magnitude:.0f}")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping...")
    capture.close()