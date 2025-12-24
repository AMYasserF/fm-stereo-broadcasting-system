"""
FM Stereo Broadcasting System - Complete Implementation
Communications Engineering Project - Cairo University

This module implements a complete FM stereo system:
- Transmitter: Stereo multiplexer -> Pre-emphasis -> FM modulator
- Receiver: FM demodulator -> De-emphasis -> Stereo decoder -> L/R recovery
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

class FMStereoParams:
    """FM Stereo System Parameters"""
    # Sampling rates
    FS_AUDIO = 44100          # Audio sample rate (Hz)
    FS_COMPOSITE = 200000     # Composite signal sample rate (Hz)
    FS_FM = 500000            # FM signal sample rate (Hz)
    
    # FM parameters
    FREQ_DEVIATION = 75000    # Frequency deviation (Hz) - default 75 kHz
    FC = 100000               # Carrier frequency for simulation (Hz)
    
    # Stereo multiplex parameters
    PILOT_FREQ = 19000        # Pilot tone frequency (Hz)
    SUBCARRIER_FREQ = 38000   # Subcarrier frequency (Hz) = 2 * pilot
    AUDIO_BW = 15000          # Audio bandwidth (Hz)
    COMPOSITE_BW = 53000      # Composite signal bandwidth (Hz)
    
    # Pre-emphasis/De-emphasis
    TAU = 75e-6               # Time constant (75 μs for US/Korea, 50 μs for Europe)
    
    # Audio duration
    DURATION = 5.0            # Audio duration in seconds

# ============================================================================
# AUDIO GENERATION
# ============================================================================

def generate_stereo_audio(duration=5.0, fs=44100):
    """
    Generate synthetic stereo audio with distinct L and R content.
    
    Left channel: Ascending frequency chirp + 440 Hz tone
    Right channel: Descending frequency chirp + 880 Hz tone
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Left channel: ascending chirp (200 Hz to 2000 Hz) + 440 Hz tone
    left_chirp = 0.3 * signal.chirp(t, f0=200, f1=2000, t1=duration, method='linear')
    left_tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    # Add some low frequency content
    left_bass = 0.2 * np.sin(2 * np.pi * 100 * t)
    left = left_chirp + left_tone + left_bass
    
    # Right channel: descending chirp (2000 Hz to 200 Hz) + 880 Hz tone
    right_chirp = 0.3 * signal.chirp(t, f0=2000, f1=200, t1=duration, method='linear')
    right_tone = 0.3 * np.sin(2 * np.pi * 880 * t)
    # Add different low frequency content
    right_bass = 0.2 * np.sin(2 * np.pi * 150 * t)
    right = right_chirp + right_tone + right_bass
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.05 * fs)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    left[:fade_samples] *= fade_in
    left[-fade_samples:] *= fade_out
    right[:fade_samples] *= fade_in
    right[-fade_samples:] *= fade_out
    
    # Normalize
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    left = left / max_val * 0.8
    right = right / max_val * 0.8
    
    return left, right, t

def save_audio(filename, left, right, fs=44100):
    """Save stereo audio to WAV file."""
    stereo = np.column_stack((left, right))
    # Handle any NaN or Inf values
    stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.8, neginf=-0.8)
    # Clip to valid range
    stereo = np.clip(stereo, -1.0, 1.0)
    stereo_int16 = np.int16(stereo * 32767)
    wavfile.write(filename, fs, stereo_int16)
    print(f"Saved audio to {filename}")

def load_audio(filename, target_fs=44100):
    """Load stereo audio from WAV file."""
    fs, data = wavfile.read(filename)
    
    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32767
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483647
    
    # Handle mono files
    if len(data.shape) == 1:
        left = right = data
    else:
        left = data[:, 0]
        right = data[:, 1]
    
    # Resample if necessary
    if fs != target_fs:
        num_samples = int(len(left) * target_fs / fs)
        left = signal.resample(left, num_samples)
        right = signal.resample(right, num_samples)
    
    t = np.linspace(0, len(left) / target_fs, len(left), endpoint=False)
    
    return left, right, t

# ============================================================================
# PRE-EMPHASIS / DE-EMPHASIS FILTERS
# ============================================================================

def design_preemphasis_filter(tau, fs):
    """
    Design pre-emphasis filter.
    Transfer function: H(s) = 1 + s*tau
    """
    # Pre-emphasis: first-order high-shelf filter
    # H(s) = (1 + s*tau) / 1
    # Using bilinear transform
    w_c = 1 / tau  # Corner frequency
    
    # Digital filter coefficients using bilinear transform
    T = 1 / fs
    alpha = 2 * tau / T
    
    b = np.array([alpha + 1, -(alpha - 1)])
    a = np.array([1, 0])
    
    # Normalize
    b = b / (alpha + 1)
    
    return b, a

def design_deemphasis_filter(tau, fs):
    """
    Design de-emphasis filter (inverse of pre-emphasis).
    Transfer function: H(s) = 1 / (1 + s*tau)
    """
    # First-order lowpass filter
    w_c = 1 / tau  # Corner frequency in rad/s
    
    # Ensure cutoff is within valid range for Butterworth filter
    cutoff_normalized = w_c / (np.pi * fs)  # Normalize to Nyquist
    cutoff_normalized = min(cutoff_normalized, 0.99)  # Keep below Nyquist
    
    # Use scipy's butter for stability
    b, a = signal.butter(1, cutoff_normalized, btype='low')
    
    return b, a

def apply_preemphasis(audio, tau, fs):
    """Apply pre-emphasis filter to audio."""
    b, a = design_preemphasis_filter(tau, fs)
    return signal.lfilter(b, a, audio)

def apply_deemphasis(audio, tau, fs):
    """Apply de-emphasis filter to audio."""
    b, a = design_deemphasis_filter(tau, fs)
    return signal.lfilter(b, a, audio)

# ============================================================================
# STEREO MULTIPLEXER (TRANSMITTER)
# ============================================================================

def stereo_multiplex(left, right, fs_audio, fs_composite, pilot_amplitude=0.1):
    """
    Create FM stereo composite signal.
    
    Components:
    - L+R (sum): 0-15 kHz baseband
    - Pilot: 19 kHz tone
    - L-R (difference): DSB-SC modulated on 38 kHz subcarrier
    
    Returns composite signal and component signals for visualization.
    """
    # Upsample audio to composite sample rate
    upsample_factor = fs_composite / fs_audio
    num_samples = int(len(left) * upsample_factor)
    
    left_up = signal.resample(left, num_samples)
    right_up = signal.resample(right, num_samples)
    
    # Time vector
    t = np.arange(num_samples) / fs_composite
    
    # Sum and difference signals
    L_plus_R = (left_up + right_up) / 2
    L_minus_R = (left_up - right_up) / 2
    
    # Pilot tone (19 kHz)
    pilot = pilot_amplitude * np.sin(2 * np.pi * FMStereoParams.PILOT_FREQ * t)
    
    # DSB-SC modulation of L-R onto 38 kHz subcarrier
    subcarrier = np.sin(2 * np.pi * FMStereoParams.SUBCARRIER_FREQ * t)
    L_minus_R_modulated = L_minus_R * subcarrier
    
    # Composite signal
    composite = L_plus_R + pilot + L_minus_R_modulated
    
    # Normalize composite signal
    composite = composite / np.max(np.abs(composite))
    
    return composite, L_plus_R, L_minus_R, L_minus_R_modulated, pilot, t

# ============================================================================
# FM MODULATOR
# ============================================================================

def fm_modulate(composite, fs_composite, fs_fm, freq_deviation, fc):
    """
    FM modulate the composite signal.
    
    FM signal: s(t) = A * cos(2*pi*fc*t + 2*pi*kf*integral(m(t)))
    where kf = freq_deviation / max(|m(t)|)
    """
    # Upsample composite to FM sample rate
    upsample_factor = fs_fm / fs_composite
    num_samples = int(len(composite) * upsample_factor)
    composite_up = signal.resample(composite, num_samples)
    
    # Time vector
    t = np.arange(num_samples) / fs_fm
    
    # Normalize message signal
    composite_norm = composite_up / np.max(np.abs(composite_up))
    
    # Frequency deviation constant
    kf = freq_deviation
    
    # Phase: integral of message signal
    phase = 2 * np.pi * kf * np.cumsum(composite_norm) / fs_fm
    
    # FM signal
    carrier = np.cos(2 * np.pi * fc * t)
    fm_signal = np.cos(2 * np.pi * fc * t + phase)
    
    return fm_signal, carrier, t, composite_up

# ============================================================================
# FM DEMODULATOR
# ============================================================================

def fm_demodulate(fm_signal, fs_fm, fs_composite, freq_deviation, fc):
    """
    FM demodulate using differentiation and envelope detection.
    
    Uses the fact that d/dt[phase] is proportional to instantaneous frequency.
    """
    # Hilbert transform to get analytic signal
    analytic_signal = signal.hilbert(fm_signal)
    
    # Instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic_signal))
    
    # Instantaneous frequency (derivative of phase)
    inst_freq = np.diff(inst_phase) * fs_fm / (2 * np.pi)
    inst_freq = np.append(inst_freq, inst_freq[-1])  # Pad to same length
    
    # Handle any NaN values from phase unwrapping
    inst_freq = np.nan_to_num(inst_freq, nan=fc, posinf=fc, neginf=fc)
    
    # Remove carrier frequency to get baseband
    demodulated = (inst_freq - fc) / freq_deviation
    
    # Low-pass filter to remove high frequency noise
    nyq = fs_fm / 2
    cutoff = FMStereoParams.COMPOSITE_BW / nyq
    b, a = signal.butter(5, cutoff, btype='low')
    demodulated = signal.filtfilt(b, a, demodulated)
    
    # Downsample to composite sample rate
    downsample_factor = fs_fm / fs_composite
    num_samples = int(len(demodulated) / downsample_factor)
    composite_recovered = signal.resample(demodulated, num_samples)
    
    # Safe normalization
    max_val = np.max(np.abs(composite_recovered))
    if max_val > 1e-10:
        composite_recovered = composite_recovered / max_val
    else:
        composite_recovered = np.zeros_like(composite_recovered)
    
    return composite_recovered

# ============================================================================
# STEREO DECODER (RECEIVER)
# ============================================================================

def extract_pilot(composite, fs, pilot_freq=19000, filter_order=8):
    """
    Extract pilot tone using bandpass filter.
    """
    # Bandpass filter around pilot frequency
    nyq = fs / 2
    low = (pilot_freq - 100) / nyq
    high = (pilot_freq + 100) / nyq
    
    b, a = signal.butter(filter_order, [low, high], btype='band')
    pilot_extracted = signal.filtfilt(b, a, composite)
    
    return pilot_extracted

def stereo_decode(composite, fs, pilot_filter_order=4, apply_deemph=True):
    """
    Decode stereo composite signal to recover L and R channels.
    
    Steps:
    1. Lowpass filter to get L+R
    2. Extract pilot tone (19 kHz)
    3. Double pilot frequency to get 38 kHz carrier (correct phase using Hilbert)
    4. Synchronous demodulation to recover L-R
    5. Matrix to recover L and R
    6. Apply de-emphasis to recovered audio (not composite!)
    
    Uses second-order sections (SOS) for numerical stability.
    """
    t = np.arange(len(composite)) / fs
    nyq = fs / 2
    
    # 1. Extract L+R (lowpass filter < 15 kHz)
    cutoff_sum = FMStereoParams.AUDIO_BW / nyq
    sos_lp = signal.butter(5, cutoff_sum, btype='low', output='sos')
    L_plus_R = signal.sosfiltfilt(sos_lp, composite)
    
    # 2. Extract pilot tone (19 kHz)
    pilot_bw = 400  # Hz bandwidth around 19 kHz
    low_pilot = (FMStereoParams.PILOT_FREQ - pilot_bw/2) / nyq
    high_pilot = (FMStereoParams.PILOT_FREQ + pilot_bw/2) / nyq
    
    sos_pilot = signal.butter(pilot_filter_order, [low_pilot, high_pilot], btype='band', output='sos')
    pilot_extracted = signal.sosfiltfilt(sos_pilot, composite)
    
    # Check for valid pilot
    pilot_max = np.max(np.abs(pilot_extracted))
    if np.isnan(pilot_max) or pilot_max < 1e-10:
        print("Warning: Pilot extraction failed, using synthetic carrier")
        carrier_38 = np.sin(2 * np.pi * FMStereoParams.SUBCARRIER_FREQ * t)
    else:
        # Normalize pilot
        pilot_normalized = pilot_extracted / pilot_max
        
        # 3. Generate 38 kHz carrier using Hilbert transform for correct phase
        # Hilbert gives analytic signal: sin(wt) + j*cos(wt)
        # Squaring: (sin + j*cos)^2 = -cos(2wt) + j*sin(2wt)
        # Take NEGATIVE imaginary part to get -sin(2wt), matching transmitter
        analytic_pilot = signal.hilbert(pilot_normalized)
        pilot_doubled = analytic_pilot ** 2
        carrier_38_raw = -np.imag(pilot_doubled)  # Negated for correct phase
        
        # Bandpass filter around 38 kHz to clean up
        sub_bw = 800  # Hz bandwidth
        low_38 = (FMStereoParams.SUBCARRIER_FREQ - sub_bw/2) / nyq
        high_38 = (FMStereoParams.SUBCARRIER_FREQ + sub_bw/2) / nyq
        sos_38 = signal.butter(4, [low_38, high_38], btype='band', output='sos')
        carrier_38 = signal.sosfiltfilt(sos_38, carrier_38_raw)
        
        # Normalize carrier
        carrier_max = np.max(np.abs(carrier_38))
        if np.isnan(carrier_max) or carrier_max < 1e-10:
            print("Warning: Carrier regeneration failed, using synthetic carrier")
            carrier_38 = np.sin(2 * np.pi * FMStereoParams.SUBCARRIER_FREQ * t)
        else:
            carrier_38 = carrier_38 / carrier_max
    
    # 4. Synchronous demodulation of L-R
    # Bandpass filter to extract 23-53 kHz (L-R DSB-SC region)
    low_lr = 23000 / nyq
    high_lr = min(53000 / nyq, 0.95)
    sos_lr = signal.butter(4, [low_lr, high_lr], btype='band', output='sos')
    L_minus_R_modulated = signal.sosfiltfilt(sos_lr, composite)
    
    # Multiply by regenerated carrier (synchronous demodulation)
    L_minus_R_demod = L_minus_R_modulated * carrier_38 * 2
    
    # Lowpass filter to get L-R baseband
    L_minus_R = signal.sosfiltfilt(sos_lp, L_minus_R_demod)
    
    # 5. Matrix decoding: L = (L+R) + (L-R), R = (L+R) - (L-R)
    left_recovered = L_plus_R + L_minus_R
    right_recovered = L_plus_R - L_minus_R
    
    # 6. Downsample to audio rate
    downsample_factor = fs / FMStereoParams.FS_AUDIO
    num_audio_samples = int(len(left_recovered) / downsample_factor)
    
    left_out = signal.resample(left_recovered, num_audio_samples)
    right_out = signal.resample(right_recovered, num_audio_samples)
    
    # 7. Apply de-emphasis to recovered BASEBAND audio (not composite!)
    if apply_deemph:
        left_out = apply_deemphasis(left_out, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
        right_out = apply_deemphasis(right_out, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    # Final safety check for NaN
    left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
    right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)
    
    return left_out, right_out, L_plus_R, L_minus_R

# ============================================================================
# NOISE FUNCTIONS
# ============================================================================

def add_awgn(signal_in, snr_db):
    """Add Additive White Gaussian Noise to achieve target SNR."""
    signal_power = np.mean(signal_in ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal_in))
    return signal_in + noise

# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def measure_bandwidth_99(signal_in, fs):
    """Measure bandwidth containing 99% of signal power."""
    # Compute power spectral density
    f, psd = signal.welch(signal_in, fs, nperseg=min(len(signal_in), 8192))
    
    # Total power
    total_power = np.sum(psd)
    
    # Find bandwidth containing 99% power
    cumulative_power = np.cumsum(psd)
    idx_99 = np.searchsorted(cumulative_power, 0.99 * total_power)
    
    bandwidth = f[min(idx_99, len(f)-1)]
    
    return bandwidth

def calculate_snr(original, recovered):
    """Calculate Signal-to-Noise Ratio in dB."""
    # Align signals (simple approach - find max correlation lag)
    min_len = min(len(original), len(recovered))
    original = original[:min_len]
    recovered = recovered[:min_len]
    
    # Normalize both signals
    original = original / np.max(np.abs(original))
    recovered = recovered / np.max(np.abs(recovered))
    
    # Scale recovered to match original
    scale = np.dot(original, recovered) / np.dot(recovered, recovered)
    recovered *= scale
    
    # Calculate SNR
    noise = original - recovered
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return 100  # Essentially perfect
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def carson_bandwidth(freq_deviation, max_modulating_freq):
    """Calculate theoretical FM bandwidth using Carson's rule."""
    # BW = 2 * (Δf + fm)
    return 2 * (freq_deviation + max_modulating_freq)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_time_frequency_plots(output_dir, left, right, t_audio, 
                                composite, L_plus_R, L_minus_R, L_minus_R_mod, pilot, t_composite,
                                fm_signal, carrier, t_fm, composite_up,
                                left_recovered, right_recovered):
    """
    Create all visualization plots with consistent time scales.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define common time window for display (first 50 ms for detail views)
    display_duration = 0.05  # 50 ms
    
    # ========== Figure 1: Input Audio L and R ==========
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
    fig1.suptitle('Input Audio - Left and Right Channels', fontsize=14, fontweight='bold')
    
    # Time domain - full signal
    ax1 = axes1[0, 0]
    ax1.plot(t_audio, left, 'b-', linewidth=0.5, label='Left')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Left Channel - Full Duration')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t_audio[-1]])
    
    ax2 = axes1[0, 1]
    ax2.plot(t_audio, right, 'r-', linewidth=0.5, label='Right')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Right Channel - Full Duration')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t_audio[-1]])
    
    # Frequency domain
    ax3 = axes1[1, 0]
    f_left, psd_left = signal.welch(left, FMStereoParams.FS_AUDIO, nperseg=4096)
    ax3.semilogy(f_left/1000, psd_left, 'b-')
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Left Channel - Spectrum')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 20])
    
    ax4 = axes1[1, 1]
    f_right, psd_right = signal.welch(right, FMStereoParams.FS_AUDIO, nperseg=4096)
    ax4.semilogy(f_right/1000, psd_right, 'r-')
    ax4.set_xlabel('Frequency (kHz)')
    ax4.set_ylabel('Power Spectral Density')
    ax4.set_title('Right Channel - Spectrum')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 20])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_input_audio.png'), dpi=150)
    plt.close()
    
    # ========== Figure 2: Composite Signal Components (Time Domain) ==========
    fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12))
    fig2.suptitle('Composite Signal Components - Time Domain', fontsize=14, fontweight='bold')
    
    # Common time range for composite signals
    t_end = min(display_duration, t_composite[-1])
    mask = t_composite <= t_end
    
    axes2[0].plot(t_composite[mask]*1000, L_plus_R[mask], 'g-', linewidth=0.8)
    axes2[0].set_ylabel('Amplitude')
    axes2[0].set_title('L+R (Sum Signal)')
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_xlim([0, t_end*1000])
    
    axes2[1].plot(t_composite[mask]*1000, L_minus_R[mask], 'm-', linewidth=0.8)
    axes2[1].set_ylabel('Amplitude')
    axes2[1].set_title('L-R (Difference Signal)')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_xlim([0, t_end*1000])
    
    axes2[2].plot(t_composite[mask]*1000, pilot[mask], 'orange', linewidth=0.8)
    axes2[2].set_ylabel('Amplitude')
    axes2[2].set_title('19 kHz Pilot Tone')
    axes2[2].grid(True, alpha=0.3)
    axes2[2].set_xlim([0, t_end*1000])
    
    axes2[3].plot(t_composite[mask]*1000, composite[mask], 'k-', linewidth=0.5)
    axes2[3].set_xlabel('Time (ms)')
    axes2[3].set_ylabel('Amplitude')
    axes2[3].set_title('Complete Composite Signal')
    axes2[3].grid(True, alpha=0.3)
    axes2[3].set_xlim([0, t_end*1000])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_composite_time.png'), dpi=150)
    plt.close()
    
    # ========== Figure 3: Composite Signal Components (Frequency Domain) ==========
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Composite Signal Components - Frequency Domain', fontsize=14, fontweight='bold')
    
    # L+R spectrum
    f, psd = signal.welch(L_plus_R, FMStereoParams.FS_COMPOSITE, nperseg=8192)
    axes3[0, 0].semilogy(f/1000, psd, 'g-')
    axes3[0, 0].set_xlabel('Frequency (kHz)')
    axes3[0, 0].set_ylabel('PSD')
    axes3[0, 0].set_title('L+R Spectrum')
    axes3[0, 0].set_xlim([0, 60])
    axes3[0, 0].grid(True, alpha=0.3)
    axes3[0, 0].axvline(x=15, color='r', linestyle='--', alpha=0.5, label='15 kHz')
    
    # L-R modulated spectrum
    f, psd = signal.welch(L_minus_R_mod, FMStereoParams.FS_COMPOSITE, nperseg=8192)
    axes3[0, 1].semilogy(f/1000, psd, 'm-')
    axes3[0, 1].set_xlabel('Frequency (kHz)')
    axes3[0, 1].set_ylabel('PSD')
    axes3[0, 1].set_title('L-R DSB-SC on 38 kHz')
    axes3[0, 1].set_xlim([0, 60])
    axes3[0, 1].grid(True, alpha=0.3)
    axes3[0, 1].axvline(x=38, color='r', linestyle='--', alpha=0.5, label='38 kHz')
    
    # Pilot spectrum
    f, psd = signal.welch(pilot, FMStereoParams.FS_COMPOSITE, nperseg=8192)
    axes3[1, 0].semilogy(f/1000, psd, 'orange')
    axes3[1, 0].set_xlabel('Frequency (kHz)')
    axes3[1, 0].set_ylabel('PSD')
    axes3[1, 0].set_title('Pilot Tone Spectrum')
    axes3[1, 0].set_xlim([0, 60])
    axes3[1, 0].grid(True, alpha=0.3)
    axes3[1, 0].axvline(x=19, color='r', linestyle='--', alpha=0.5, label='19 kHz')
    
    # Complete composite spectrum
    f, psd = signal.welch(composite, FMStereoParams.FS_COMPOSITE, nperseg=8192)
    axes3[1, 1].semilogy(f/1000, psd, 'k-')
    axes3[1, 1].set_xlabel('Frequency (kHz)')
    axes3[1, 1].set_ylabel('PSD')
    axes3[1, 1].set_title('Complete Composite Spectrum')
    axes3[1, 1].set_xlim([0, 60])
    axes3[1, 1].grid(True, alpha=0.3)
    axes3[1, 1].axvline(x=15, color='g', linestyle='--', alpha=0.5)
    axes3[1, 1].axvline(x=19, color='orange', linestyle='--', alpha=0.5)
    axes3[1, 1].axvline(x=38, color='m', linestyle='--', alpha=0.5)
    axes3[1, 1].axvline(x=53, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_composite_spectrum.png'), dpi=150)
    plt.close()
    
    # ========== Figure 4: FM Carrier Before and After Modulation ==========
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('FM Carrier - Before and After Modulation', fontsize=14, fontweight='bold')
    
    # Show exactly 5 complete carrier cycles for clear visualization
    # At 100 kHz carrier, one cycle = 10 μs, so 5 cycles = 50 μs
    num_cycles = 5
    cycle_period = 1.0 / FMStereoParams.FC  # Period of one carrier cycle
    display_time = num_cycles * cycle_period  # Total display time
    
    # Calculate samples - use enough samples for smooth curves
    samples_per_cycle = int(FMStereoParams.FS_FM * cycle_period)
    total_samples = num_cycles * samples_per_cycle
    
    # Create synchronized display starting from t=0
    t_display = t_fm[:total_samples]
    carrier_display = carrier[:total_samples]
    fm_display = fm_signal[:total_samples]
    
    # Carrier before modulation (time domain)
    axes4[0, 0].plot(t_display*1e6, carrier_display, 'b-', linewidth=1.5)
    axes4[0, 0].set_xlabel('Time (μs)')
    axes4[0, 0].set_ylabel('Amplitude')
    axes4[0, 0].set_title(f'Unmodulated Carrier ({num_cycles} cycles at {FMStereoParams.FC/1000:.0f} kHz)')
    axes4[0, 0].grid(True, alpha=0.3)
    axes4[0, 0].set_ylim([-1.2, 1.2])
    axes4[0, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes4[0, 0].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    
    # FM signal after modulation (time domain)
    axes4[0, 1].plot(t_display*1e6, fm_display, 'r-', linewidth=1.5)
    axes4[0, 1].set_xlabel('Time (μs)')
    axes4[0, 1].set_ylabel('Amplitude')
    axes4[0, 1].set_title(f'FM Modulated Signal (constant amplitude, varying frequency)')
    axes4[0, 1].grid(True, alpha=0.3)
    axes4[0, 1].set_ylim([-1.2, 1.2])
    axes4[0, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes4[0, 1].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    
    # Carrier spectrum
    f, psd = signal.welch(carrier, FMStereoParams.FS_FM, nperseg=8192)
    axes4[1, 0].semilogy(f/1000, psd, 'b-')
    axes4[1, 0].set_xlabel('Frequency (kHz)')
    axes4[1, 0].set_ylabel('PSD')
    axes4[1, 0].set_title('Unmodulated Carrier Spectrum')
    axes4[1, 0].set_xlim([50, 150])
    axes4[1, 0].grid(True, alpha=0.3)
    axes4[1, 0].axvline(x=FMStereoParams.FC/1000, color='r', linestyle='--', alpha=0.5)
    
    # FM signal spectrum
    f, psd = signal.welch(fm_signal, FMStereoParams.FS_FM, nperseg=8192)
    axes4[1, 1].semilogy(f/1000, psd, 'r-')
    axes4[1, 1].set_xlabel('Frequency (kHz)')
    axes4[1, 1].set_ylabel('PSD')
    axes4[1, 1].set_title('FM Modulated Signal Spectrum')
    axes4[1, 1].set_xlim([0, 250])
    axes4[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_fm_carrier.png'), dpi=150)
    plt.close()
    
    # ========== Figure 5: Recovered Audio ==========
    t_recovered = np.linspace(0, len(left_recovered)/FMStereoParams.FS_AUDIO, 
                               len(left_recovered), endpoint=False)
    
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 8))
    fig5.suptitle('Recovered Audio - Left and Right Channels', fontsize=14, fontweight='bold')
    
    # Time domain
    axes5[0, 0].plot(t_recovered, left_recovered, 'b-', linewidth=0.5)
    axes5[0, 0].set_xlabel('Time (s)')
    axes5[0, 0].set_ylabel('Amplitude')
    axes5[0, 0].set_title('Recovered Left Channel')
    axes5[0, 0].grid(True, alpha=0.3)
    axes5[0, 0].set_xlim([0, t_recovered[-1]])
    
    axes5[0, 1].plot(t_recovered, right_recovered, 'r-', linewidth=0.5)
    axes5[0, 1].set_xlabel('Time (s)')
    axes5[0, 1].set_ylabel('Amplitude')
    axes5[0, 1].set_title('Recovered Right Channel')
    axes5[0, 1].grid(True, alpha=0.3)
    axes5[0, 1].set_xlim([0, t_recovered[-1]])
    
    # Frequency domain
    f_left, psd_left = signal.welch(left_recovered, FMStereoParams.FS_AUDIO, nperseg=4096)
    axes5[1, 0].semilogy(f_left/1000, psd_left, 'b-')
    axes5[1, 0].set_xlabel('Frequency (kHz)')
    axes5[1, 0].set_ylabel('PSD')
    axes5[1, 0].set_title('Recovered Left Spectrum')
    axes5[1, 0].grid(True, alpha=0.3)
    axes5[1, 0].set_xlim([0, 20])
    
    f_right, psd_right = signal.welch(right_recovered, FMStereoParams.FS_AUDIO, nperseg=4096)
    axes5[1, 1].semilogy(f_right/1000, psd_right, 'r-')
    axes5[1, 1].set_xlabel('Frequency (kHz)')
    axes5[1, 1].set_ylabel('PSD')
    axes5[1, 1].set_title('Recovered Right Spectrum')
    axes5[1, 1].grid(True, alpha=0.3)
    axes5[1, 1].set_xlim([0, 20])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_recovered_audio.png'), dpi=150)
    plt.close()
    
    # ========== Figure 6: Input vs Recovered Comparison ==========
    fig6, axes6 = plt.subplots(2, 2, figsize=(14, 8))
    fig6.suptitle('Input vs Recovered Audio Comparison', fontsize=14, fontweight='bold')
    
    # Align lengths for comparison
    min_len = min(len(left), len(left_recovered))
    
    # Left channel comparison
    axes6[0, 0].plot(t_audio[:min_len], left[:min_len], 'b-', linewidth=0.5, alpha=0.7, label='Original')
    axes6[0, 0].plot(t_recovered[:min_len], left_recovered[:min_len], 'g--', linewidth=0.5, alpha=0.7, label='Recovered')
    axes6[0, 0].set_xlabel('Time (s)')
    axes6[0, 0].set_ylabel('Amplitude')
    axes6[0, 0].set_title('Left Channel Comparison')
    axes6[0, 0].legend()
    axes6[0, 0].grid(True, alpha=0.3)
    axes6[0, 0].set_xlim([0, t_audio[min_len-1]])
    
    # Right channel comparison
    axes6[0, 1].plot(t_audio[:min_len], right[:min_len], 'r-', linewidth=0.5, alpha=0.7, label='Original')
    axes6[0, 1].plot(t_recovered[:min_len], right_recovered[:min_len], 'orange', linewidth=0.5, alpha=0.7, label='Recovered')
    axes6[0, 1].set_xlabel('Time (s)')
    axes6[0, 1].set_ylabel('Amplitude')
    axes6[0, 1].set_title('Right Channel Comparison')
    axes6[0, 1].legend()
    axes6[0, 1].grid(True, alpha=0.3)
    axes6[0, 1].set_xlim([0, t_audio[min_len-1]])
    
    # Zoomed view - first 100 ms
    zoom_samples = int(0.1 * FMStereoParams.FS_AUDIO)
    axes6[1, 0].plot(t_audio[:zoom_samples]*1000, left[:zoom_samples], 'b-', linewidth=1, label='Original')
    axes6[1, 0].plot(t_recovered[:zoom_samples]*1000, left_recovered[:zoom_samples], 'g--', linewidth=1, label='Recovered')
    axes6[1, 0].set_xlabel('Time (ms)')
    axes6[1, 0].set_ylabel('Amplitude')
    axes6[1, 0].set_title('Left Channel - First 100 ms')
    axes6[1, 0].legend()
    axes6[1, 0].grid(True, alpha=0.3)
    
    axes6[1, 1].plot(t_audio[:zoom_samples]*1000, right[:zoom_samples], 'r-', linewidth=1, label='Original')
    axes6[1, 1].plot(t_recovered[:zoom_samples]*1000, right_recovered[:zoom_samples], 'orange', linewidth=1, label='Recovered')
    axes6[1, 1].set_xlabel('Time (ms)')
    axes6[1, 1].set_ylabel('Amplitude')
    axes6[1, 1].set_title('Right Channel - First 100 ms')
    axes6[1, 1].legend()
    axes6[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_comparison.png'), dpi=150)
    plt.close()
    
    print(f"All plots saved to {output_dir}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_fm_stereo_system(output_dir="output", freq_deviation=75000, input_snr=None):
    """
    Run the complete FM stereo system.
    
    Parameters:
    - output_dir: Directory to save outputs
    - freq_deviation: FM frequency deviation in Hz
    - input_snr: Input SNR in dB (None for no noise)
    
    Returns:
    - Dictionary with all signals and measurements
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FM STEREO BROADCASTING SYSTEM")
    print("="*60)
    
    # ====== STEP 1: Generate/Load Audio ======
    print("\n[1/6] Generating stereo audio...")
    left, right, t_audio = generate_stereo_audio(
        duration=FMStereoParams.DURATION, 
        fs=FMStereoParams.FS_AUDIO
    )
    
    # Save input audio
    save_audio(os.path.join(output_dir, "input_stereo.wav"), left, right, FMStereoParams.FS_AUDIO)
    
    # ====== STEP 2: Apply Pre-emphasis ======
    print("[2/6] Applying pre-emphasis...")
    left_preemph = apply_preemphasis(left, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    right_preemph = apply_preemphasis(right, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    # ====== STEP 3: Stereo Multiplex ======
    print("[3/6] Creating stereo composite signal...")
    composite, L_plus_R, L_minus_R, L_minus_R_mod, pilot, t_composite = stereo_multiplex(
        left_preemph, right_preemph,
        FMStereoParams.FS_AUDIO,
        FMStereoParams.FS_COMPOSITE
    )
    
    # ====== STEP 4: FM Modulation ======
    print("[4/6] FM modulating...")
    fm_signal, carrier, t_fm, composite_up = fm_modulate(
        composite,
        FMStereoParams.FS_COMPOSITE,
        FMStereoParams.FS_FM,
        freq_deviation,
        FMStereoParams.FC
    )
    
    # Add noise if specified
    if input_snr is not None:
        print(f"      Adding AWGN (SNR = {input_snr} dB)...")
        fm_signal = add_awgn(fm_signal, input_snr)
    
    # ====== STEP 5: FM Demodulation ======
    print("[5/6] FM demodulating...")
    composite_recovered = fm_demodulate(
        fm_signal,
        FMStereoParams.FS_FM,
        FMStereoParams.FS_COMPOSITE,
        freq_deviation,
        FMStereoParams.FC
    )
    
    # ====== STEP 6: Stereo Decode ======
    # Note: stereo_decode now handles de-emphasis internally after decoding
    print("[6/6] Decoding stereo...")
    left_recovered, right_recovered, L_plus_R_rec, L_minus_R_rec = stereo_decode(
        composite_recovered,
        FMStereoParams.FS_COMPOSITE
    )
    
    # Normalize outputs
    max_out = max(np.max(np.abs(left_recovered)), np.max(np.abs(right_recovered)))
    if max_out > 0:
        left_recovered = left_recovered / max_out * 0.8
        right_recovered = right_recovered / max_out * 0.8
    
    # Save output audio
    save_audio(os.path.join(output_dir, "output_stereo.wav"), left_recovered, right_recovered, 
               FMStereoParams.FS_AUDIO)
    
    # ====== Create Visualizations ======
    print("\nGenerating visualizations...")
    create_time_frequency_plots(
        os.path.join(output_dir, "figures"),
        left, right, t_audio,
        composite, L_plus_R, L_minus_R, L_minus_R_mod, pilot, t_composite,
        fm_signal, carrier, t_fm, composite_up,
        left_recovered, right_recovered
    )
    
    # ====== Calculate Metrics ======
    print("\nCalculating metrics...")
    
    # SNR
    snr_left = calculate_snr(left, left_recovered)
    snr_right = calculate_snr(right, right_recovered)
    
    # Bandwidth
    measured_bw = measure_bandwidth_99(fm_signal, FMStereoParams.FS_FM)
    theoretical_bw = carson_bandwidth(freq_deviation, FMStereoParams.COMPOSITE_BW)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Frequency Deviation: {freq_deviation/1000:.0f} kHz")
    print(f"Measured Bandwidth (99% power): {measured_bw/1000:.1f} kHz")
    print(f"Theoretical Bandwidth (Carson): {theoretical_bw/1000:.1f} kHz")
    print(f"Output SNR (Left):  {snr_left:.1f} dB")
    print(f"Output SNR (Right): {snr_right:.1f} dB")
    print("="*60)
    
    # Return results dictionary
    results = {
        'left_input': left,
        'right_input': right,
        'left_output': left_recovered,
        'right_output': right_recovered,
        'composite': composite,
        'fm_signal': fm_signal,
        'snr_left': snr_left,
        'snr_right': snr_right,
        'measured_bw': measured_bw,
        'theoretical_bw': theoretical_bw,
        'freq_deviation': freq_deviation
    }
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the FM stereo system with default parameters
    results = run_fm_stereo_system(output_dir="output", freq_deviation=75000)
    
    print("\n[DONE] FM Stereo System completed successfully!")
    print("Check the 'output' folder for:")
    print("  - input_stereo.wav  (synthetic input audio)")
    print("  - output_stereo.wav (recovered output audio)")
    print("  - figures/          (visualization plots)")
