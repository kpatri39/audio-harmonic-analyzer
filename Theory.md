# Mathematical Theory: Real-Time FFT Audio Analysis

## Table of Contents

1. [Introduction](#introduction)
2. [Fourier Transform Fundamentals](#fourier-transform-fundamentals)
3. [Fast Fourier Transform Algorithm](#fast-fourier-transform-algorithm)
4. [Windowing and Spectral Leakage](#windowing-and-spectral-leakage)
5. [Musical Acoustics and Equal Temperament](#musical-acoustics-and-equal-temperament)
6. [Parabolic Interpolation for Sub-Bin Accuracy](#parabolic-interpolation-for-sub-bin-accuracy)
7. [Peak Detection and Discrete Calculus](#peak-detection-and-discrete-calculus)
8. [Limitations and Challenges](#limitations-and-challenges)
9. [Real-World Instrument Acoustics](#real-world-instrument-acoustics)
10. [Conclusion](#conclusion)

---

## Introduction

This document provides a comprehensive mathematical foundation for the FFT-based audio analyzer. We explore the theory behind frequency-domain analysis, pitch detection algorithms, and the fundamental limitations of computational audio processing.

The analyzer implements real-time pitch detection using the Fast Fourier Transform (FFT), achieving sub-bin frequency accuracy through parabolic interpolation while maintaining computational efficiency suitable for live audio processing.

---

## Fourier Transform Fundamentals

### Continuous Fourier Transform

The **Fourier Transform** decomposes a continuous time-domain signal $f(t)$ into its frequency components. The forward transform is defined as:

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt$$

where:
- $f(t)$ is the time-domain signal
- $F(\omega)$ is the frequency-domain representation
- $\omega = 2\pi f$ is the angular frequency
- $i = \sqrt{-1}$ is the imaginary unit

The **inverse Fourier Transform** recovers the time-domain signal:

$$f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} \, d\omega$$

### Euler's Formula

The complex exponential in the Fourier Transform is understood through **Euler's formula**:

$$e^{i\theta} = \cos\theta + i\sin\theta$$

This reveals that the Fourier Transform decomposes signals into sums of cosines and sines - the fundamental building blocks of periodic functions.

### Discrete Fourier Transform (DFT)

For digital audio (discrete samples), we use the **Discrete Fourier Transform**:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi kn/N}$$

where:
- $x_n$ are the $N$ time-domain samples
- $X_k$ are the $N$ frequency-domain coefficients
- $k$ is the frequency bin index (0 to N-1)
- $n$ is the time sample index

Each $X_k$ is a **complex number** encoding both magnitude and phase:

$$X_k = a_k + ib_k$$

The **magnitude** (strength of frequency $k$) is:

$$|X_k| = \sqrt{a_k^2 + b_k^2}$$

The **phase** (timing offset) is:

$$\phi_k = \arctan\left(\frac{b_k}{a_k}\right)$$

For pitch detection, we primarily use magnitude and discard phase information.

### Frequency Bin Mapping

Each bin $k$ corresponds to a physical frequency:

$$f_k = \frac{k \cdot f_s}{N}$$

where:
- $f_s$ is the sampling rate (e.g., 44,100 Hz)
- $N$ is the number of samples

**Example:** With $N = 4096$ samples at $f_s = 44100$ Hz:
- Frequency resolution: $\Delta f = 44100/4096 \approx 10.77$ Hz
- Bin 41: $f_{41} = 41 \times 10.77 = 441.57$ Hz (near A4)

---

## Fast Fourier Transform Algorithm

### Computational Complexity

The naive DFT requires $O(N^2)$ operations - computing each of $N$ frequency bins requires summing over all $N$ time samples.

The **Fast Fourier Transform** (FFT), discovered by Cooley and Tukey in 1965, reduces this to $O(N \log N)$ through a divide-and-conquer approach.

### Cooley-Tukey Algorithm

The FFT exploits symmetry in the DFT computation. For $N = 2^m$ (power of 2), we can split the DFT into even and odd indexed samples:

$$X_k = \sum_{n=0}^{N/2-1} x_{2n} e^{-i 2\pi k(2n)/N} + \sum_{n=0}^{N/2-1} x_{2n+1} e^{-i 2\pi k(2n+1)/N}$$

This separates into two DFTs of size $N/2$:

$$X_k = E_k + e^{-i 2\pi k/N} O_k$$

where $E_k$ and $O_k$ are the DFTs of the even and odd samples.

Recursively applying this split reduces complexity from $O(N^2)$ to $O(N \log N)$.

### Real FFT Optimization

Since audio signals are real-valued (not complex), we use `rfft` (Real FFT) which exploits **conjugate symmetry**:

$$X_{N-k} = \overline{X_k}$$

This means the second half of the spectrum mirrors the first half, so we only compute the first $N/2 + 1$ coefficients, saving 50% of computation.

---

## Windowing and Spectral Leakage

### The Leakage Problem

When we capture a finite chunk of audio, we're effectively multiplying the infinite signal by a rectangular window:

$$x_{windowed}[n] = x[n] \cdot w_{rect}[n]$$

where $w_{rect}[n] = 1$ for $0 \leq n < N$ and $0$ elsewhere.

**Multiplication in time domain = Convolution in frequency domain**

The rectangular window's frequency response is a **sinc function**, which has significant sidelobes. These sidelobes cause energy from one frequency to "leak" into neighboring bins, creating false frequency components.

### Hann Window

The **Hann window** tapers the signal smoothly to zero at the edges:

$$w_{Hann}[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$

Properties:
- Starts and ends at 0 (smooth edges)
- Peak of 1 at center
- Bell-shaped curve

The Hann window's frequency response has much smaller sidelobes than the rectangular window, reducing spectral leakage at the cost of slightly wider main lobe (reduced frequency resolution).

**Trade-off:** 
- Rectangular window: Best frequency resolution, worst leakage
- Hann window: Good compromise between resolution and leakage

### Implementation
```python
windowed_data = audio_data * np.hanning(len(audio_data))
```

Each sample is multiplied by its corresponding window value, smoothly tapering the edges to zero.

---

## Musical Acoustics and Equal Temperament

### The Equal Temperament System

Western music uses **equal temperament**, where each octave is divided into 12 semitones with equal frequency ratios.

The frequency ratio between adjacent semitones is:

$$r = 2^{1/12} \approx 1.05946$$

Starting from a reference pitch (A4 = 440 Hz), any note $n$ semitones away has frequency:

$$f_n = 440 \cdot 2^{n/12}$$

**Examples:**
- A4: $n = 0$, $f = 440$ Hz
- A#4: $n = 1$, $f = 440 \cdot 2^{1/12} = 466.16$ Hz
- A5: $n = 12$, $f = 440 \cdot 2^{12/12} = 880$ Hz (one octave up)

### Inverse: Frequency to Note

Given a frequency $f$, we can find the number of semitones from A4:

$$n = 12 \log_2\left(\frac{f}{440}\right)$$

The octave is:

$$\text{octave} = 4 + \left\lfloor \frac{n}{12} \right\rfloor$$

The note within the octave:

$$\text{note\_index} = (n + 9) \bmod 12$$

where the +9 offset accounts for A being at index 9 in the chromatic scale starting from C.

### Harmonic Series

When a string vibrates, it produces not just the fundamental frequency $f_0$, but also **harmonics** (overtones) at integer multiples:

$$f_n = n \cdot f_0, \quad n = 1, 2, 3, 4, \ldots$$

**Example for A4 (440 Hz):**
- 1st harmonic (fundamental): 440 Hz
- 2nd harmonic: 880 Hz (A5, one octave up)
- 3rd harmonic: 1320 Hz (E6, fifth above)
- 4th harmonic: 1760 Hz (A6, two octaves up)

The relative strength of these harmonics determines the **timbre** - why a violin sounds different from a flute even at the same pitch.

---

## Parabolic Interpolation for Sub-Bin Accuracy

### The Problem

With $N = 4096$ samples at 44.1 kHz, frequency resolution is ~10.77 Hz per bin. A note at 440 Hz might fall between bins at 430.66 Hz and 441.43 Hz, causing ~10 Hz detection error.

### The Solution

**Parabolic interpolation** estimates the "true" peak frequency between bins by fitting a parabola through three points around the detected peak.

### Mathematical Derivation

Given a peak at bin $k$ with magnitudes:
- $\alpha = |X_{k-1}|$ (left neighbor)
- $\beta = |X_k|$ (peak)
- $\gamma = |X_{k+1}|$ (right neighbor)

We fit a parabola $p(x) = ax^2 + bx + c$ through points $(-1, \alpha)$, $(0, \beta)$, $(1, \gamma)$.

The parabola's peak occurs where its derivative is zero:

$$p'(x) = 2ax + b = 0$$

Solving for the peak location:

$$x_{peak} = -\frac{b}{2a}$$

Using the three points to solve for $a$ and $b$:

$$a = \frac{\alpha - 2\beta + \gamma}{2}$$

$$b = \frac{\gamma - \alpha}{2}$$

Therefore:

$$\delta = x_{peak} = -\frac{b}{2a} = \frac{\alpha - \gamma}{2(\alpha - 2\beta + \gamma)}$$

This can be simplified to:

$$\delta = \frac{0.5(\alpha - \gamma)}{\alpha - 2\beta + \gamma}$$

The interpolated frequency is:

$$f_{true} = f_k + \delta \cdot \Delta f$$

where $\Delta f$ is the frequency bin spacing.

### Practical Considerations

**Bounds checking:** The denominator $\alpha - 2\beta + \gamma$ can be very small if the peak is flat, causing numerical instability. We check:
```python
if abs(denominator) > 1e-10:
    delta = 0.5 * (alpha - gamma) / denominator
```

**Clamping:** To prevent wild overshoots from noisy data, we clamp $\delta$:
```python
delta = np.clip(delta, -0.5, 0.5)
```

This ensures the interpolated frequency stays within half a bin of the detected peak.

### Accuracy Improvement

With parabolic interpolation:
- Bin spacing: 10.77 Hz
- Actual accuracy: ~1-2 Hz

This provides **5-10× improvement** in frequency precision without increasing chunk size!

---

## Peak Detection and Discrete Calculus

### Continuous Peak Detection

In continuous calculus, a local maximum occurs at $x = x_0$ if:

1. **First derivative is zero:** $f'(x_0) = 0$
2. **Second derivative is negative:** $f''(x_0) < 0$

This is the **second derivative test** for maxima.

### Discrete Approximations

For discrete signals, we approximate derivatives using **finite differences**.

**First derivative** (central difference):

$$f'[i] \approx \frac{f[i+1] - f[i-1]}{2}$$

**Second derivative**:

$$f''[i] \approx f[i+1] - 2f[i] + f[i-1]$$

A peak at index $i$ satisfies:
- $f'[i] \approx 0$ (or changes sign from positive to negative)
- $f''[i] < 0$ (concave down)

### SciPy's find_peaks

The `scipy.signal.find_peaks` function implements sophisticated peak detection with additional criteria:

**Prominence:** Measures how much a peak "stands out" from the surrounding baseline. Defined as the vertical distance between the peak and its lowest contour line.

$$\text{prominence} = \text{peak\_height} - \min(\text{left\_baseline}, \text{right\_baseline})$$

**Height:** Absolute magnitude threshold - peaks below this are ignored.

**Distance:** Minimum spacing between peaks (in samples) to avoid detecting multiple peaks for the same feature.

### Our Implementation
```python
peaks, properties = signal.find_peaks(
    magnitudes,
    prominence=max_mag * 0.15,  # Must stand out by 15% of max
    height=max_mag * 0.3,        # Must be at least 30% of max
    distance=10                  # At least 10 bins apart
)
```

This filters the spectrum to significant peaks while suppressing noise and closely-spaced false detections.

---

## Limitations and Challenges

### 1. Harmonic Ambiguity

**The Problem:** When analyzing musical notes, harmonics can be as strong or stronger than the fundamental frequency.

**Example:** Playing violin G3 (196 Hz):
- Fundamental: 196 Hz (often weak)
- 2nd harmonic: 392 Hz (often stronger due to instrument resonance)

Peak detection finds 392 Hz → incorrectly reports G4 instead of G3.

**Mathematical Issue:** There is no way to distinguish between:
- A strong 2nd harmonic of 196 Hz
- An actual note at 392 Hz

Both produce the same spectral peak!

### 2. Octave Confusion

When two notes are played an octave apart (e.g., A4 at 440 Hz and A5 at 880 Hz), the 2nd harmonic of A4 is exactly 880 Hz - identical to the fundamental of A5.

The spectrum shows:
- Peak at 440 Hz
- Peak at 880 Hz

**Question:** Is 880 Hz:
- A harmonic of 440 Hz? (only one note playing)
- A separate note? (two notes playing)

**No definitive answer** from frequency analysis alone!

### 3. Low-Frequency Detection Challenges

Low-pitched instruments face several issues:

**Weak fundamentals:** Low frequencies have longer wavelengths and may not excite the instrument body's resonances as effectively.

**Harmonic density:** More harmonics fit within the audible range, creating a crowded spectrum.

**Frequency resolution:** With fixed window size, low frequencies have proportionally coarser resolution:
- At 200 Hz: 10.77 Hz error = 5.4% error
- At 2000 Hz: 10.77 Hz error = 0.54% error

### 4. The Missing Fundamental Phenomenon

The human auditory system can perceive a pitch even when the fundamental frequency is absent! If harmonics at 880, 1320, and 1760 Hz are present, we hear 440 Hz even if that frequency isn't in the signal.

**Implication:** Simple peak detection fails to match human pitch perception in these cases.

### 5. Polyphonic Detection (Multi-Note Analysis)

Detecting multiple simultaneous notes is fundamentally harder:

**Overlapping Harmonics:** Each note produces a harmonic series. These series overlap and interact:
- D4 (294 Hz): 294, 588, 882, 1176 Hz...
- A4 (440 Hz): 440, 880, 1320 Hz...
- Notice: 880 Hz and 882 Hz are nearly identical!

**Magnitude Competition:** If one note is louder, its harmonics can mask another note's fundamental.

**Combinatorial Complexity:** With $n$ detected peaks, determining which combinations form valid harmonic series is an NP-hard optimization problem.

**Attempted Solution (Commented in Code):**

We implemented a multi-note detector that:
1. Finds multiple peaks using `find_peaks`
2. Filters out harmonics by checking if peaks are integer multiples
3. Uses magnitude comparison to distinguish octaves from harmonics

**Results:** 
- Works for simple cases (clean double stops)
- Fails for octaves (2nd harmonic = next note's fundamental)
- Produces false positives with complex timbres
- Missing notes when one is much quieter

This remains an active research area in Music Information Retrieval (MIR).

### 6. Frequency-Time Uncertainty

There's a fundamental trade-off governed by the **Heisenberg Uncertainty Principle** analog for signal processing:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

**Implications:**
- **Large window (high $N$):** Better frequency resolution, slower updates
- **Small window (low $N$):** Faster updates, coarser frequency resolution

Our choice of $N = 4096$ at 44.1 kHz:
- Time window: ~93 ms
- Frequency resolution: ~10.77 Hz
- Update rate: ~11 Hz

This balances real-time responsiveness with acceptable frequency precision for musical applications.

---

## Real-World Instrument Acoustics

### Violin String Characteristics

Through empirical testing, we discovered instrument-specific behaviors that reveal the connection between mathematics and physical acoustics:

**E String (659 Hz) - "Bright" Character:**
- The E string sometimes shows **competing peaks** between fundamental and upper harmonics
- FFT occasionally locks onto harmonics instead of fundamental
- **Mathematical manifestation of "brightness":** Upper harmonics have comparable magnitude to fundamental
- The "whistling" phenomenon: Specific harmonics become so dominant they're audible as separate pitches
- **Hypothesis:** Steel core and high tension emphasize high-frequency content; violin body resonances may align with E string harmonics

**G String (196 Hz) - Weak Fundamental:**
- Consistently detects 392 Hz (2nd harmonic) instead of 196 Hz (fundamental)
- **Physical cause:** Low frequencies don't efficiently couple to the violin body
- The fundamental exists but is significantly weaker than the 2nd harmonic
- **This is not an algorithm error** - it reflects the actual acoustic output of the instrument

**D and A Strings - Balanced Response:**
- More reliable detection with fundamentals clearly dominating
- Middle range of the instrument has more balanced harmonic structure
- These strings show the "mellow" character the violin maker described

### Bow Position and Technique

The FFT analyzer reveals how playing technique affects harmonic content:

**Bow closer to bridge (sul ponticello):**
- Spectrum shows enhanced high-frequency harmonics
- Many strong peaks in the 1000-2000 Hz range
- Creates the characteristic "glassy" or "metallic" tone

**Bow over fingerboard (sul tasto):**
- Fundamental dominates the spectrum
- Fewer and weaker harmonics visible
- Creates a "soft" or "flute-like" tone

**Mathematical interpretation:** Bowing position acts as a mechanical **filter**, emphasizing different modal vibrations of the string.

### Microphone Frequency Response

Testing with different microphones revealed:

**Built-in MacBook microphone:**
- Emphasis on mid-range frequencies (500-2000 Hz)
- Roll-off below 100 Hz and above 10 kHz
- Spectrum shows artificially boosted 2nd-4th harmonics

**External condenser microphone (hypothetical):**
- Flatter frequency response
- Better low-frequency capture
- More accurate representation of fundamental frequencies

This demonstrates that **the observed spectrum depends on both the instrument AND the capture device** - a reminder that all measurements are mediated by their tools.

---

## Conclusion

This project demonstrates the power and limitations of FFT-based audio analysis. We've successfully implemented:

✅ **Real-time frequency analysis** with sub-bin accuracy  
✅ **Musical pitch detection** using equal temperament mathematics  
✅ **Visualization of acoustic phenomena** including harmonic series and timbre  
✅ **Understanding of fundamental limitations** in polyphonic and low-frequency detection

### Key Takeaways

1. **The FFT is a powerful but imperfect tool** - it provides a mathematical window into sound, but physical acoustics are complex

2. **Parabolic interpolation bridges the gap** between computational efficiency and frequency precision

3. **Harmonic ambiguity is fundamental** - you cannot always distinguish harmonics from separate notes without additional context

4. **Real-world instruments reveal mathematical principles** - the spectrum visualizes what "bright" vs "mellow" actually means

5. **Trade-offs are inevitable** - frequency resolution vs. time resolution, computational cost vs. accuracy, simplicity vs. sophistication

### Applications Beyond Music

The techniques developed here apply to:
- Speech recognition and voice analysis
- Seismology (earthquake frequency analysis)
- Radar and sonar signal processing
- Telecommunications (frequency modulation)
- Medical imaging (MRI, ultrasound)

The mathematics of Fourier analysis is universal - this project demonstrates its application to a domain where we can directly hear and verify the results.

### Future Research Directions

Advanced topics not implemented but worth exploring:

**Autocorrelation-based pitch detection:** An alternative to FFT that can better handle missing fundamentals

**Constant-Q Transform:** Variable-resolution transform that matches musical octave structure

**Deep learning approaches:** Neural networks trained on large datasets can learn to distinguish harmonics from fundamentals

**Phase vocoder:** Analyzes phase evolution over time for enhanced pitch tracking

**Source separation:** Decomposing polyphonic audio into individual notes/instruments

Each of these represents decades of ongoing research in digital signal processing and music information retrieval.

---

## References

1. Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297-301.

2. Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. *Proceedings of the IEEE*, 66(1), 51-83.

3. Smith, J. O. (2011). *Spectral Audio Signal Processing*. W3K Publishing.

4. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.

5. Rabiner, L. R., & Schafer, R. W. (2011). *Theory and Applications of Digital Speech Processing*. Pearson.