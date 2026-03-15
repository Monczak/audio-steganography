# Audio Steganography - Project Plan

---

## Literature Review

### [1] Baseline - Comparative Benchmarking of Classical Techniques
**Comparative Analysis of Audio Steganography Methods**
DergiPark / University of Twente BA thesis (essay.utwente.nl/95988)

The foundational quantitative reference for the project. Benchmarks LSB, Spread Spectrum, Cepstrum/Echo Hiding, and Wavelet Coding on a standardised dataset across SNR, capacity, and robustness. The key finding is a strict inverse correlation between payload size and signal quality across all methods, and a clear robustness ranking: DSSS > Echo Hiding > Wavelet > LSB. The SNR figures from this source (LSB ~97 dB, DSSS ~44 dB, Echo Hiding ~28 dB) are used directly in the project's comparison section. Also surfaces the counterintuitive "legacy vs. neural" paradox: classical spread spectrum outperforms most GAN-based approaches under additive noise.

### [2] Phase Coding - Improved Mid-Frequency Adaptive Algorithm
**An Improved Phase Coding Audio Steganography Algorithm**
arXiv 2408.13277 (2024)

Extends the classical Fourier phase coding method by dynamically segmenting the signal and targeting the mid-frequency range (roughly 1-8 kHz) for embedding, rather than encoding everything into the first block's absolute phase. The motivation is psychoacoustic: low frequencies carry the bulk of perceptible energy and should not be disturbed; high frequencies are aggressively discarded by lossy compression. By concentrating the payload in mid-frequency phase components, the algorithm achieves better BER under compression and lower steganalysis detectability than the classical single-block approach. Directly informs Section 2 of the notebook.

### [3] Spectrogram Steganography - Chaotic STFT/DWT Image Concealment
**A robust audio steganography technique based on image encryption using different chaotic maps**
PMC / NCBI (PMC11436861, 2024)

Describes a system for hiding a full image inside an audio file by injecting chaotic-map-encrypted pixel data into the high-frequency STFT coefficients of the carrier. The chaotic map pre-scrambles the image's pixel distribution before embedding, so that even if an adversary locates the modified coefficients, the data is visually meaningless without the decryption key. Reports PSNR exceeding 91.2 dB - essentially inaudible. This is the rigorous version of the simpler spectrogram painting approach and is the main reference for Section 3.

### [4] Echo Hiding - Cepstrum Extraction and Improved Kernels
**Presenting a Method for Improving Echo Hiding**
ResearchGate (349335429)

Addresses the two main weaknesses of classical echo hiding: bipolar echo kernels that produce audible artifacts, and susceptibility to cepstral steganalysis. Proposes hybrid kernels and chaotic pseudo-random key generation to vary the echo delay parameters across segments, defeating fixed-threshold cepstral detectors. The paper also provides the cepstrum-based extraction algorithm - inverse FFT of the log power spectrum, then autocorrelation peak detection - that the notebook implements for decoding. Establishes the critical perceptual threshold: delays of 1-3 ms are perceived as natural reverberation; beyond 3 ms the echo becomes audible as a distinct artifact.

### [5] Spread Spectrum - DSSS and Gaussian Cyclostationarity Camouflage
**Gaussian-Distributed Spread-Spectrum for Covert Communications**
PMC / NIH (PMC10144851, 2024)

Covers both the standard DSSS embedding mechanism (bipolar modulation with a pseudo-random chip sequence) and the more advanced GDSS extension. The key vulnerability of standard DSSS is cyclostationarity - the periodic chip sequence leaves a detectable pattern in cyclic spectral analysis even if the message cannot be read. GDSS addresses this by mathematically reshaping the transmitted signal to follow a Gaussian amplitude distribution, making it statistically indistinguishable from thermal noise. Directly motivates the steganalysis experiment in Section 5: the cyclic spectral detector catches standard DSSS but would be defeated by GDSS.

### [6] MP3 Bitstream - Huffman Table Transformation
**High capacity reversible data hiding in MP3 based on Huffman table transformation**
AIMS Press / Mathematical Biosciences and Engineering (DOI: 10.3934/mbe.2019158)

The technical reference for the most elegant technique in the project. The MP3 standard defines multiple Huffman tables for entropy coding quantised MDCT coefficients; the steganographic algorithm exploits the fact that paired tables produce acoustically equivalent results, using the table selection itself as the information channel. Zero acoustic distortion (no sample values are changed), fully reversible, and the payload is extracted by reading frame header metadata rather than decoding the audio. A Python library (`mp3-steganography-lib`) implements this directly, making it the only technique in the project that operates natively on a compressed MP3 without a decode/re-encode cycle.

### [7] Deep Learning Ceiling - FGS-Audio Adversarial Framework
**FGS-Audio: Fixed-Decoder Framework for Audio Steganography with Adversarial Perturbation Generation**
ResearchGate (2025)

Included as a "ceiling" reference rather than an implementation target. FGS-Audio integrates a psychoacoustic masking model directly into gradient descent - the network computes per-frequency inaudibility thresholds and constrains adversarial perturbations to remain beneath them. Training includes simulated distortion layers (MP3 compression, filtering, noise) so the decoder learns invariant features. Not feasible to implement from scratch, but its architecture motivates the extensions section and defines the theoretical ideal against which classical methods are honestly compared.

---

## Overview

The project is structured as a single Jupyter notebook implementing five steganographic techniques of progressively increasing sophistication, followed by a steganalysis section that attempts to detect each one, and a final comparison. The goal is to understand the embedding and extraction mechanics hands-on rather than treating techniques as black boxes.

The core structure per technique is:

```
Cover audio + Secret payload
        \/
  Embedding algorithm
        \/
  Stego audio file
        \/
  Extraction algorithm -> Recovered payload
        \/
  Quality metrics (SNR, BER, capacity)
        \/
  Steganalysis attempt - can we detect it?
```

---

## Environment & Dependencies

| Library | Purpose |
|---|---|
| `numpy`, `scipy` | FFT, IFFT, signal processing, correlation |
| `librosa` | Audio loading, STFT, spectrogram display |
| `soundfile` | PCM read/write for WAV I/O |
| `pydub` | Audio manipulation and format conversion |
| `matplotlib` | Waveform, spectrogram, and metric plots |
| `Pillow` | Image loading/saving for image-in-audio techniques |
| `wave` | Raw WAV frame access for LSB |
| `mp3-steganography-lib` | Huffman table swapping for MP3 |
| `ffmpeg` (subprocess) | Compression attack simulation |

---

## Part 1 - Baseline: LSB Steganography

**Difficulty:** Low. **Value:** High - establishes the vocabulary and metrics for everything that follows.

LSB is the entry point. It is not the most interesting technique, but implementing it cleanly sets up the whole notebook: it introduces the encode/decode loop, the SNR measurement, the BER calculation, and the capacity formula that every subsequent section reuses.

### 1.1 Implementation

Using the `wave` module for direct byte access:
- Read all audio frames into a mutable byte array
- Encode text or binary payload by replacing the LSB of each sample byte (bitwise AND to zero it, OR to inject the payload bit)
- Prepend a fixed-length header encoding payload length so the decoder knows when to stop
- Decode by extracting LSBs sequentially and reconstructing the byte stream

Extend to 2-bit and 3-bit variants: overwrite the two or three least significant bits per sample. Plot SNR vs. bits-used-per-sample to make the capacity/quality tradeoff concrete.

### 1.2 Steganalysis: LSB Distribution Test

The LSB distribution of natural audio is not perfectly random - the least significant bit of audio samples correlates weakly with the sample above it. Embedding a uniformly random payload disrupts this correlation and drives the LSB histogram toward 50/50. Implement a chi-squared test on the LSB histogram before and after embedding. This is the core of StegoScan-style heuristic detection - fast to implement and produces a clear pass/fail result.

### 1.3 Robustness Test

Apply an MP3 compression round-trip (WAV -> MP3 -> WAV via FFmpeg, then re-extract). The payload will be destroyed. This is not a flaw to fix - it is an honest demonstration of why more sophisticated techniques exist, and it motivates everything in Parts 2-5.

---

## Part 2 - Phase Coding

**Difficulty:** Medium. **Value:** High - introduces frequency-domain thinking and the psychoacoustic principle of phase deafness.

### 2.1 Classical Phase Coding

Implement the standard algorithm in numpy:
1. Segment the audio into N non-overlapping blocks of length L
2. Apply FFT to each block; extract magnitude and phase matrices
3. Compute inter-block phase differences: Δφᵢ = φᵢ - φᵢ₋₁
4. Map payload bits to phase values: `0 -> π/2`, `1 -> -π/2`
5. Inject the encoded phase into block 0 only; enforce DFT symmetry for real-valued output
6. Propagate original Δφ differences forward through all subsequent blocks to maintain phase continuity
7. Reconstruct via IFFT

Extraction: the receiver takes block 0, runs FFT, and checks whether the phase at each target bin is closer to π/2 or -π/2.

The key limitation to demonstrate explicitly: because the payload is entirely in block 0, capacity is capped at L/2 bits (half the block size, due to conjugate symmetry). Increasing L for more capacity begins to warp inter-block phase relations and introduces audible artifacts.

### 2.2 Improved Mid-Frequency Variant (arXiv 2408.13277)

Extend the classical approach following the 2024 paper: instead of embedding only in block 0, embed one chunk of payload per block, but restrict the modified frequency bins to the mid-frequency range (e.g., 1-8 kHz). Below that range, phase changes are perceptible because those frequencies dominate the perceived signal. Above it, phase shifts survive compression poorly. Plot which frequency bins are modified to make the targeting concrete. Compare BER after an MP3 round-trip against the classical single-block approach - the mid-frequency version degrades more gracefully.

---

## Part 3 - Spectrogram Steganography (Image in Audio)

**Difficulty:** Medium. **Value:** High - visually the most striking result in the notebook, and conceptually unlike all other techniques here.

This section hides a grayscale image inside an audio file by synthesising the image's pixel values as frequency amplitudes over time. The resulting audio sounds like noise but renders the image when viewed as a spectrogram.

### 3.1 Basic Spectrogram Painting

- Load a grayscale image; scale it to target dimensions (width = time steps, height = frequency bins)
- For each column (time step), construct a frequency-domain vector where each bin's amplitude equals the corresponding pixel's brightness
- Apply inverse FFT per column to get a time-domain audio segment
- Concatenate all segments into the final synthesised audio

The audio sounds jarring in isolation. Play it and show the spectrogram - the image appears immediately. This is the most visually rewarding single result in the project.

### 3.2 Psychoacoustic Camouflage

Spectrogram painting is only useful if the resulting signal can be hidden under real audio. Rather than a left/right split - which is immediately obvious to any listener wearing headphones, since one channel would sound like a fax machine - use a mid-side (M/S) representation:

- Mid = (L + R) / 2 - the mono-compatible sum, carrying all dominant musical energy
- Side = (L - R) / 2 - the stereo difference signal, carrying only spatial width information

The side channel already contains legitimate low-level content in any real stereo recording (instrument panning, room ambience), so injecting additional signal into it is far less suspicious. On mono playback - phone speakers, cheap Bluetooth - the side channel cancels out entirely ((M+S) + (M-S) = 2M), making the hidden content completely inaudible on those systems.
Implementation:
1. Decode the stereo carrier to L and R arrays
2. Compute M = (L + R) / 2 and S = (L - R) / 2
3. Generate the image-audio signal (from Section 3.1), frequency-shifted to 13-19 kHz and gain-reduced by 30-40 dB
4. Add the image signal to the S channel
5. Reconstruct L = M + S and R = M - S and write the output file

Show three spectrograms side by side: the original carrier, the stego output (which should look essentially identical), and the isolated S channel (which reveals the image). Also verify the mono cancellation property explicitly by summing L + R and showing the image disappears.

---

## Part 4 - Echo Hiding

**Difficulty:** Medium-High. **Value:** Good - introduces cepstrum analysis, which is both the decoding primitive and the main steganalysis attack, making for an elegant self-contained section.

### 4.1 Echo Kernel Implementation

Segment the audio into blocks (typical block size: 1000-4000 samples). For each block:
- Convolve with an echo kernel parameterised by delay d and decay α: `h(t) = δ(t) + α·δ(t - d)`
- Use delay d₀ (e.g. 1.0 ms) to encode binary `0`, delay d₁ (e.g. 1.5 ms) to encode binary `1`
- Apply a smoothing mixer at block boundaries to avoid audible clicks (linear cross-fade between adjacent kernel responses)

Keep α ≤ 0.5 and delays strictly within the 1-3 ms perceptual window established by source [4]. Show the waveform before and after - the difference should be essentially invisible.

### 4.2 Cepstrum Extraction

To decode:
1. Compute the power spectrum of each block: `|FFT(x)|²`
2. Take the log - this converts the multiplicative echo structure into an additive one
3. Apply inverse FFT to yield the cepstrum
4. Compute the autocorrelation of the cepstrum; identify the dominant peak at short quefrency
5. Compare the peak location to d₀ vs. d₁ thresholds and output the corresponding bit

This is the most DSP-intensive extraction step in the notebook. Plot the cepstrum for a block encoded with each delay value - the peak shifts visibly and measurably between d₀ and d₁.

### 4.3 Steganalysis: Cepstral Peak Detector

The same cepstrum analysis used for extraction is also the standard steganalysis attack. On a clean audio file, the cepstrum has no strong periodic peak at short quefrency. On an echo-encoded file, a clear peak appears at the echo delay. Implement a simple detector: flag any block where the cepstrum autocorrelation exceeds a threshold at sub-5 ms quefrency. Quantify the false positive rate on clean audio vs. detection rate on stego audio. Note explicitly that the improved hybrid kernels from source [4] were designed to defeat exactly this detector.

---

## Part 5 - Spread Spectrum (DSSS)

**Difficulty:** Medium-High. **Value:** High - the most robust classical technique, and the steganalysis experiment (cyclostationarity detection) is genuinely interesting and replicates current research.

### 5.1 DSSS Embedding

Following the standard mechanism:
- Map payload bits to bipolar values: `0 -> -1`, `1 -> +1`
- Generate a pseudo-random chip sequence using a seeded PRNG (the seed is the shared key); chip rate >> message bit rate (e.g., 100:1 ratio)
- Multiply the bipolar message by the chip sequence to produce the spread signal
- Scale to a target amplitude ε (typically 0.001-0.01 relative to peak audio amplitude)
- Add to the cover audio: `stego = cover + ε · (message ⊗ chip)`

The spread signal is buried below the noise floor and is perceptually invisible at low ε.

### 5.2 Correlation Extraction

Recovery does not require knowing the original audio:
- Multiply the stego audio by the same chip sequence (the receiver must know the PRNG seed)
- Integrate over chip-length windows - the chip is orthogonal to the cover audio, so the cover contribution averages toward zero, while the payload contribution coherently accumulates
- Threshold the correlation output: positive -> `1`, negative -> `0`

Demonstrate graceful degradation: increase ε, add noise, apply MP3 compression. BER rises slowly rather than jumping catastrophically, unlike LSB. This directly illustrates why DSSS is the most robust entry in the comparison table.

### 5.3 Steganalysis: Cyclic Spectral Analysis

Standard DSSS has a detectable footprint: the repeating chip sequence creates cyclostationary periodicity in the signal's second-order statistics, visible as a peak in the cyclic spectral density at the chip repetition frequency. Implement a simplified cyclic autocorrelation estimator:
- Compute the cyclic autocorrelation of the stego signal at the chip repetition period
- A clean audio file shows no significant peak; a DSSS-stego file shows a clear one

This directly replicates the attack that the GDSS paper (source [5]) was designed to defeat. Note that GDSS would render this detector ineffective by reshaping the amplitude distribution - a natural extension for Section 8.

---

## Part 6 - MP3 Huffman Table Swapping

**Difficulty:** Low (the library handles the heavy lifting). **Value:** High - conceptually the most elegant technique, and the only one that operates on a compressed MP3 without touching audio samples at all.

### 6.1 Implementation via mp3-steganography-lib

The `mp3-steganography-lib` library exposes a facade over the Huffman swapping logic:
- Instantiate the encoder with a WAV source and a target bitrate
- Call `encode_wav_to_mp3(message)` - the library converts to MP3 while embedding by selecting Huffman table variants according to payload bits at each frame
- Call `decode_mp3(stego_file)` on the receiver side to extract the message by reading frame header metadata, without decoding the audio at all

### 6.2 What Makes This Structurally Different

Demonstrate two properties that no other technique here can claim:

**Zero acoustic distortion:** Compare the spectrogram of the Huffman-stego MP3 to a straight encode of the same WAV. They are identical - the audio samples are unchanged, only the compression metadata differs. SNR is theoretically infinite.

**Reversibility:** Extract the payload, restore the optimal Huffman tables, compare the result to a clean encode - they match. The steganographic footprint can be completely removed, leaving no forensic trace in the bitstream.

Contrast this with the LSB MP3 round-trip from Part 1: LSB embeds into WAV samples and is destroyed by the same compression that Huffman swapping uses as its carrier.

---

## Part 7 - Comparison and Summary

### 7.1 Metrics Table

Run all five techniques on the same carrier file with the same payload and populate a comparison table, using the SNR benchmarks from source [1] as reference points:

| Technique | Avg SNR (dB) | Capacity | BER after MP3 roundtrip | Detectable? |
|---|---|---|---|---|
| LSB (1-bit) | ~97 | Very high | 100% (destroyed) | Yes - chi^2 test |
| Phase Coding | High | Low | Moderate | Hard |
| Spectrogram (STFT/DWT) | ~91 | Medium | Low | Low (visual only) |
| Echo Hiding | ~28 | Low-Medium | Moderate | Yes - cepstral peak |
| DSSS | ~44 | Moderate | Low | Partial - cyclostat. |
| Huffman Swapping | Infinite (0 distortion) | Medium | 0% (native to MP3) | Structural only |

### 7.2 Capacity/Quality Trade-off Plot

Plot SNR vs. payload size for LSB (varying bits-per-sample), phase coding (varying block size), and DSSS (varying ε and chip rate). This makes the steganographic triad - imperceptibility, capacity, robustness - concrete and visual rather than abstract.

### 7.3 Robustness Ladder

Test all techniques under the same attack sequence and record BER at each stage:
1. Clean extraction (no attack)
2. Additive white Gaussian noise at -40 dB
3. Low-pass filter at 8 kHz
4. MP3 re-encode at 128 kbps

DSSS should survive steps 1-3; Huffman swapping survives all four; LSB fails at step 2. This ladder tells the complete story of why the technique landscape is not a single-winner problem.

---

## Possible Extensions (If Time Allows)

**GDSS Implementation** - the Gaussian-distributed countermeasure to the cyclostationary detector from source [5]. Requires reshaping the spread signal's amplitude distribution before addition to the carrier; mathematically tractable and would directly demonstrate that the steganalysis from Section 5.3 can be defeated.

**AAC_SIGN** - the AAC equivalent of Huffman swapping: flip the sign bits of low-amplitude QMDCT coefficients. Requires parsing AAC bitstream headers, which is more involved than MP3 due to AAC's more complex frame structure. No ready library equivalent to mp3-steganography-lib exists.

**Image payload via DSSS** - extend the spread spectrum section to hide a full greyscale image by treating the flattened pixel array as a bitstream, then reconstructing and displaying the recovered image. The correlation-based extraction already generalises; it only requires a quantisation and reshape step at the output.

**Adaptive phase coding** - implement the per-band frequency sensitivity sweep (inject perturbations per frequency bin, measure phase detectability) to empirically identify which bins are least perceptible to both the HAS and steganalysis, and target those preferentially.

---

## Recommended Notebook Structure

```
Section 0 - Setup & imports
            Helper functions: snr(), ber(), capacity_bits(), plot_spectrogram()
Section 1 - LSB
  1.1  Encode and decode text payload in WAV
  1.2  SNR vs bits-per-sample sweep
  1.3  LSB distribution steganalysis (chi^2 test)
  1.4  MP3 round-trip robustness test -> BER = 100%
Section 2 - Phase Coding
  2.1  Classical FFT phase coding (block 0 only)
  2.2  Mid-frequency adaptive variant (arXiv 2408.13277)
  2.3  BER comparison under compression
Section 3 - Spectrogram Steganography
  3.1  Basic image-to-spectrogram synthesis
  3.2  Stereo channel psychoacoustic camouflage
  3.3  Chaotic STFT/DWT embedding and extraction (PMC11436861)
  3.4  PSNR measurement vs. paper benchmark
Section 4 - Echo Hiding
  4.1  Echo kernel construction and block embedding
  4.2  Cepstrum-based extraction
  4.3  Steganalysis: cepstral peak detector + false positive rate
Section 5 - Spread Spectrum (DSSS)
  5.1  Bipolar encoding and chip spreading
  5.2  Correlation-based extraction
  5.3  BER under noise and compression (graceful degradation)
  5.4  Steganalysis: cyclic spectral density estimator
Section 6 - MP3 Huffman Table Swapping
  6.1  Encode and decode via mp3-steganography-lib
  6.2  Zero-distortion verification (spectrogram comparison)
  6.3  Reversibility demonstration
Section 7 - Comparison
  7.1  Metrics table
  7.2  Capacity/quality trade-off plots
  7.3  Robustness ladder across all techniques
Section 8 - Extensions (placeholder cells)
```
