# pylint: disable=all
# type: ignore

import time
import numpy as np
from audio_capture import AudioCapture

capture = AudioCapture()

print("\nPlay your violin! Press Ctrl+C to stop.\n")

try:
    while True:
        results = capture.analyze_chunk()
        
        if results['fundamental'] > 0:
            fund_freq = results['fundamental']
            freqs = results['frequencies']
            mags = results['magnitudes']
            
            idx = np.argmin(np.abs(freqs - fund_freq))
            fundamental_magnitude = mags[idx]
            
            if fundamental_magnitude > 9:
                print(f"Note: {results['note']:4s} | Frequency: {results['fundamental']:7.2f} Hz | Mag: {fundamental_magnitude:.0f}")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping...")
    capture.close()