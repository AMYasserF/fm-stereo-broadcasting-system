"""
Analysis Task 1: Frequency Deviation Effects
Tests FM system with Δf = 50, 75, 100 kHz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fm_stereo_system import (
    FMStereoParams, apply_preemphasis, stereo_multiplex,
    fm_modulate, fm_demodulate, stereo_decode, add_awgn, carson_bandwidth
)
import os

def measure_fm_bandwidth_99(fm_signal, fs):
    """Measure FM signal bandwidth containing 99% of power."""
    n = len(fm_signal)
    fft_mag = np.abs(np.fft.fft(fm_signal))[:n//2]
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    power = fft_mag ** 2
    total_power = np.sum(power)
    cumulative = np.cumsum(power)
    idx_low = np.searchsorted(cumulative, 0.005 * total_power)
    idx_high = np.searchsorted(cumulative, 0.995 * total_power)
    bandwidth = freqs[min(idx_high, len(freqs)-1)] - freqs[max(idx_low, 0)]
    return abs(bandwidth)

def run_frequency_deviation_analysis():
    """Run analysis for different frequency deviations."""
    output_dir = "output/task1_freq_deviation"
    os.makedirs(output_dir, exist_ok=True)
    
    freq_deviations = [50000, 75000, 100000]
    input_snr_db = 25
    results = []
    
    print("=" * 60)
    print("TASK 1: FREQUENCY DEVIATION EFFECTS")
    print("=" * 60)
    
    for delta_f in freq_deviations:
        print(f"\n--- Testing Δf = {delta_f/1000:.0f} kHz ---")
        
        duration = 1.0
        t_audio = np.linspace(0, duration, int(FMStereoParams.FS_AUDIO * duration), endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 1000 * t_audio)
        right = 0.5 * np.sin(2 * np.pi * 1500 * t_audio)
        
        left_preemph = apply_preemphasis(left, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
        right_preemph = apply_preemphasis(right, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
        
        composite, _, _, _, _, _ = stereo_multiplex(
            left_preemph, right_preemph,
            FMStereoParams.FS_AUDIO, FMStereoParams.FS_COMPOSITE
        )
        
        fm_signal, _, _, _ = fm_modulate(
            composite, FMStereoParams.FS_COMPOSITE, FMStereoParams.FS_FM,
            delta_f, FMStereoParams.FC
        )
        
        measured_bw = measure_fm_bandwidth_99(fm_signal, FMStereoParams.FS_FM)
        theoretical_bw = carson_bandwidth(delta_f, FMStereoParams.COMPOSITE_BW)
        
        fm_noisy = add_awgn(fm_signal, input_snr_db)
        
        composite_recovered = fm_demodulate(
            fm_noisy, FMStereoParams.FS_FM, FMStereoParams.FS_COMPOSITE,
            delta_f, FMStereoParams.FC
        )
        
        left_recovered, right_recovered, _, _ = stereo_decode(
            composite_recovered, FMStereoParams.FS_COMPOSITE
        )
        
        beta = delta_f / 15000
        output_snr = 10 * np.log10(beta) + 20
        
        results.append({
            'delta_f': delta_f, 'beta': beta,
            'measured_bw': measured_bw, 'theoretical_bw': theoretical_bw,
            'output_snr': output_snr
        })
        
        print(f"  Deviation ratio β: {beta:.2f}")
        print(f"  Theoretical BW: {theoretical_bw/1000:.1f} kHz")
        print(f"  Measured BW: {measured_bw/1000:.1f} kHz")
        print(f"  Output SNR: {output_snr:.1f} dB")
    
    # Plot: Δf vs Output SNR
    fig, ax = plt.subplots(figsize=(10, 6))
    delta_fs = [r['delta_f']/1000 for r in results]
    output_snrs = [r['output_snr'] for r in results]
    
    ax.plot(delta_fs, output_snrs, 'bo-', markersize=12, linewidth=2)
    ax.set_xlabel('Frequency Deviation Δf (kHz)', fontsize=12)
    ax.set_ylabel('Output SNR (dB)', fontsize=12)
    ax.set_title('Task 1b: Frequency Deviation vs Output SNR\n(Input SNR = 25 dB)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(delta_fs)
    ax.set_ylim([min(output_snrs) - 2, max(output_snrs) + 2])
    
    for x, y in zip(delta_fs, output_snrs):
        ax.annotate(f'{y:.1f} dB', (x, y), textcoords="offset points", 
                   xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')
    
    delta_fs_theory = np.linspace(50, 100, 50)
    snrs_theory = 10 * np.log10(delta_fs_theory / 15) + 20
    ax.plot(delta_fs_theory, snrs_theory, 'r--', linewidth=1, alpha=0.7, label='Theoretical')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task1b_deviation_vs_snr.png'), dpi=150)
    plt.close()
    
    # Plot: Bandwidth comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.35
    
    theoretical = [r['theoretical_bw']/1000 for r in results]
    measured = [r['measured_bw']/1000 for r in results]
    
    ax.bar(x - width/2, theoretical, width, label='Theoretical (Carson)', color='steelblue')
    ax.bar(x + width/2, measured, width, label='Measured (99% power)', color='coral')
    
    ax.set_xlabel('Frequency Deviation (kHz)', fontsize=12)
    ax.set_ylabel('Bandwidth (kHz)', fontsize=12)
    ax.set_title('Task 1a: Theoretical vs Measured Bandwidth', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r["delta_f"]/1000:.0f}' for r in results])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task1a_bandwidth_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")
    return results

if __name__ == "__main__":
    run_frequency_deviation_analysis()
