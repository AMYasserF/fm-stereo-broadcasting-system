"""
Analysis Task 3: Channel Separation Analysis
Measures stereo channel separation using left-only test tone
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fm_stereo_system import (
    FMStereoParams, apply_preemphasis, stereo_multiplex,
    fm_modulate, fm_demodulate, stereo_decode
)
import os

def measure_separation_rms(left, right):
    """Measure separation using RMS ratio."""
    l_rms = np.sqrt(np.mean(left**2))
    r_rms = np.sqrt(np.mean(right**2))
    if r_rms < 1e-12:
        return 60.0
    return np.clip(20 * np.log10(l_rms / r_rms), -10, 60)

def measure_separation_at_freq(left, right, freq, fs):
    """Measure separation at specific frequency using FFT."""
    n = len(left)
    freqs = np.fft.fftfreq(n, 1/fs)
    idx = np.argmin(np.abs(freqs - freq))
    
    left_fft = np.abs(np.fft.fft(left))
    right_fft = np.abs(np.fft.fft(right))
    
    l_power = left_fft[idx]
    r_power = right_fft[idx]
    
    if r_power < 1e-12:
        return 60.0
    return np.clip(20 * np.log10(l_power / r_power), -10, 60)

def run_channel_separation_analysis():
    """Run channel separation analysis."""
    output_dir = "output/task3_channel_separation"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("TASK 3: CHANNEL SEPARATION ANALYSIS")
    print("=" * 60)
    
    duration = 1.0
    t = np.linspace(0, duration, int(FMStereoParams.FS_AUDIO * duration), endpoint=False)
    
    # Left-only test tone
    left_orig = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right_orig = np.zeros_like(t)
    
    left_pre = apply_preemphasis(left_orig, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    right_pre = apply_preemphasis(right_orig, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    composite, L_plus_R, L_minus_R, L_minus_R_mod, pilot, t_comp = stereo_multiplex(
        left_pre, right_pre, FMStereoParams.FS_AUDIO, FMStereoParams.FS_COMPOSITE
    )
    
    fm_signal, _, _, _ = fm_modulate(
        composite, FMStereoParams.FS_COMPOSITE, FMStereoParams.FS_FM,
        FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
    )
    
    composite_rec = fm_demodulate(
        fm_signal, FMStereoParams.FS_FM, FMStereoParams.FS_COMPOSITE,
        FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
    )
    
    left_rec, right_rec, L_plus_R_rec, L_minus_R_rec = stereo_decode(
        composite_rec, FMStereoParams.FS_COMPOSITE
    )
    
    sep_rms = measure_separation_rms(left_rec, right_rec)
    sep_freq = measure_separation_at_freq(left_rec, right_rec, 1000, FMStereoParams.FS_AUDIO)
    
    print(f"\nSeparation (RMS): {sep_rms:.1f} dB")
    print(f"Separation (1 kHz): {sep_freq:.1f} dB")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 3: Channel Separation Analysis', fontsize=14, fontweight='bold')
    
    # Time domain
    t_plot = t[:2000]
    axes[0, 0].plot(t_plot * 1000, left_rec[:2000], 'b-', linewidth=1, label='Left')
    axes[0, 0].plot(t_plot * 1000, right_rec[:2000], 'r-', linewidth=1, alpha=0.7, label='Right')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Recovered Audio (Left-Only Input)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency domain
    f_left, psd_left = signal.welch(left_rec, FMStereoParams.FS_AUDIO, nperseg=2048)
    f_right, psd_right = signal.welch(right_rec, FMStereoParams.FS_AUDIO, nperseg=2048)
    
    axes[0, 1].semilogy(f_left/1000, psd_left + 1e-20, 'b-', linewidth=1.5, label='Left')
    axes[0, 1].semilogy(f_right/1000, psd_right + 1e-20, 'r-', linewidth=1.5, alpha=0.7, label='Right')
    axes[0, 1].set_xlabel('Frequency (kHz)')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].set_title('Audio Spectrum')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 5])
    
    # Composite spectrum
    f_comp, psd_comp = signal.welch(composite, FMStereoParams.FS_COMPOSITE, nperseg=4096)
    axes[1, 0].semilogy(f_comp/1000, psd_comp + 1e-20, 'g-', linewidth=1)
    axes[1, 0].axvline(x=19, color='r', linestyle='--', alpha=0.5, label='Pilot (19 kHz)')
    axes[1, 0].axvline(x=38, color='orange', linestyle='--', alpha=0.5, label='Subcarrier (38 kHz)')
    axes[1, 0].set_xlabel('Frequency (kHz)')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_title('Composite Signal Spectrum')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 55])
    
    # Summary
    axes[1, 1].axis('off')
    summary = f"""
    CHANNEL SEPARATION RESULTS
    ==========================
    
    Test Signal: 1 kHz tone in left channel only
    
    Measurements:
    • RMS-based separation: {sep_rms:.1f} dB
    • Frequency-specific (1 kHz): {sep_freq:.1f} dB
    
    Interpretation:
    • Separation > 20 dB: Excellent stereo
    • Separation 15-20 dB: Good stereo
    • Separation < 15 dB: Poor separation
    
    Result: {'GOOD' if sep_rms > 15 else 'POOR'} stereo separation
    """
    axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                    fontsize=11, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task3_channel_separation.png'), dpi=150)
    plt.close()
    
    print(f"\nPlot saved to {output_dir}")
    return sep_rms, sep_freq

if __name__ == "__main__":
    run_channel_separation_analysis()
