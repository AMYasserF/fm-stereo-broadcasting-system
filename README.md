# FM Stereo Broadcasting System

A complete Python implementation of an FM stereo broadcasting system with transmitter and receiver components, including comprehensive analysis of system performance.

## Overview

This project simulates the complete FM stereo broadcasting chain used in commercial FM radio:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Stereo     │───>│  FM          │───>│  Channel    │───>│  FM          │
│  Audio      │    │  Transmitter │    │  (AWGN)     │    │  Receiver    │
│  (L, R)     │    │              │    │             │    │  (L', R')    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

## Features

- **Pre-emphasis / De-emphasis** filtering (75 μs time constant)
- **Stereo Multiplexing** with 19 kHz pilot tone and 38 kHz DSB-SC subcarrier
- **FM Modulation / Demodulation** with configurable frequency deviation
- **Pilot-based Carrier Recovery** using Hilbert transform
- **AWGN Channel Simulation** for noise analysis

## System Parameters

| Parameter | Value |
|-----------|-------|
| Audio Sample Rate | 44.1 kHz |
| Composite Sample Rate | 200 kHz |
| FM Sample Rate | 500 kHz |
| Carrier Frequency | 100 kHz |
| Frequency Deviation | 75 kHz |
| Pilot Frequency | 19 kHz |
| Subcarrier Frequency | 38 kHz |

## Project Structure

```
├── fm_stereo_system.py           # Core FM stereo system implementation
├── analysis_task1_frequency_deviation.py   # Task 1: Frequency deviation effects
├── analysis_task2_noise_immunity.py        # Task 2: Noise immunity analysis
├── analysis_task3_channel_separation.py    # Task 3: Channel separation
├── analysis_task4_filter_design.py         # Task 4: Filter design impact
├── analysis_task5_robustness.py            # Task 5: Pilot error robustness
└── output/                                 # Generated plots and results
```

## Requirements

```
numpy
scipy
matplotlib
```

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Usage

### Run Complete System Demo
```python
from fm_stereo_system import fm_stereo_demo
results = fm_stereo_demo()
```

### Run Individual Analysis Tasks
```bash
python analysis_task1_frequency_deviation.py
python analysis_task2_noise_immunity.py
python analysis_task3_channel_separation.py
python analysis_task4_filter_design.py
python analysis_task5_robustness.py
```

## Analysis Tasks

### Task 1: Frequency Deviation Effects
Tests the system with Δf = 50, 75, 100 kHz to analyze:
- Carson's rule bandwidth vs measured bandwidth
- Output SNR improvement with frequency deviation

**Results:**
| Δf (kHz) | Deviation Ratio β | Output SNR |
|----------|-------------------|------------|
| 50 | 3.33 | 25.2 dB |
| 75 | 5.00 | 27.0 dB |
| 100 | 6.67 | 28.2 dB |

### Task 2: Noise Immunity
Tests system at different input SNR levels (5, 10, 15, 20, 25 dB):
- FM SNR transfer characteristic
- Channel separation degradation with noise

### Task 3: Channel Separation
Measures stereo separation using left-only test tone:
- **RMS-based separation:** 18.5 dB
- **Frequency-specific (1 kHz):** 18.6 dB

### Task 4: Filter Design Impact
Compares pilot extraction filter orders (4, 8, 12):

| Filter Order | Channel Separation |
|--------------|-------------------|
| 4 | 18.7 dB (Best) |
| 8 | 15.6 dB |
| 12 | 14.5 dB |

**Finding:** Lower filter orders preserve pilot waveform better for Hilbert-based carrier recovery.

### Task 5: System Robustness
Tests pilot frequency errors from -500 Hz to +500 Hz:
- Separation at 0 Hz offset: 18.4 dB
- Tolerance for ≥15 dB separation: ±100 Hz
- At ±500 Hz offset: Complete stereo failure (~0 dB)

## Key Implementation Details

### Stereo Multiplexing
The composite baseband signal follows the standard FM stereo format:
```
Composite = (L+R) + pilot + (L-R)·sin(2·38kHz·t)
```

Where:
- **L+R (0-15 kHz):** Mono-compatible sum signal
- **Pilot (19 kHz):** Phase reference for decoding
- **L-R (23-53 kHz):** DSB-SC modulated difference signal

### Carrier Recovery
Uses Hilbert transform for phase-coherent 38 kHz carrier regeneration:
```python
analytic = hilbert(pilot)
carrier_38 = -imag(analytic ** 2)  # Frequency doubling with correct phase
```

### De-emphasis
Applied to recovered audio (not composite) to match transmitter pre-emphasis:
```
H(s) = 1 / (1 + s·τ)  where τ = 75 μs
```

## Output Examples

The analysis scripts generate plots in the `output/` directory:

- `task1_freq_deviation/` - Bandwidth and SNR vs frequency deviation
- `task2_noise_immunity/` - SNR transfer and separation vs noise
- `task3_channel_separation/` - Time/frequency domain analysis
- `task4_filter_design/` - Filter response and separation comparison
- `task5_robustness/` - Separation vs pilot frequency error

## Theory

### FM Improvement
FM provides SNR improvement over AM due to the "capture effect":
```
SNR_out ≈ 3β² · SNR_in  (above threshold)
```

### Carson's Bandwidth Rule
```
BW = 2(Δf + f_m)
```

For FM stereo with Δf = 75 kHz and f_m = 53 kHz:
```
BW = 2(75 + 53) = 256 kHz
```

## License

MIT License

## References

1. B.P. Lathi, "Modern Digital and Analog Communication Systems"
2. FCC FM Stereo Standard (1961)
3. ITU-R BS.450-3 - Transmission standards for FM sound broadcasting
