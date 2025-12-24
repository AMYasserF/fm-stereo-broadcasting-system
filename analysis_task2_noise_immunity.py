"""
Analysis Task 2: Noise Immunity Analysis
Tests FM system at different input SNR levels: 5, 10, 15, 20, 25 dB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fm_stereo_system import (
    FMStereoParams, apply_preemphasis, stereo_multiplex,
    fm_modulate, fm_demodulate, stereo_decode, add_awgn
)
import os

def calculate_snr(original, recovered):
    """Calculate SNR between original and recovered signals."""
    min_len = min(len(original), len(recovered))
    orig = original[:min_len]
    rec = recovered[:min_len]
    
    rec_max = np.max(np.abs(rec))
    if rec_max > 0:
        rec = rec / rec_max
    
    orig_max = np.max(np.abs(orig))
    if orig_max > 0:
        orig = orig / orig_max
    
    if np.dot(rec, rec) > 0:
        scale = np.dot(orig, rec) / np.dot(rec, rec)
        rec = rec * scale
    
    noise = orig - rec
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return 50.0
    return np.clip(10 * np.log10(signal_power / noise_power), -10, 50)

def measure_separation(left, right):
    """Measure channel separation in dB."""
    l_rms = np.sqrt(np.mean(left**2))
    r_rms = np.sqrt(np.mean(right**2))
    if r_rms < 1e-12:
        return 60.0
    if l_rms < 1e-12:
        return 0.0
    return np.clip(20 * np.log10(l_rms / r_rms), -10, 60)

def run_noise_immunity_analysis():
    """Run noise immunity analysis."""
    output_dir = "output/task2_noise_immunity"
    os.makedirs(output_dir, exist_ok=True)
    
    input_snrs = [5, 10, 15, 20, 25]
    results = []
    
    print("=" * 60)
    print("TASK 2: NOISE IMMUNITY ANALYSIS")
    print("=" * 60)
    
    duration = 1.0
    t = np.linspace(0, duration, int(FMStereoParams.FS_AUDIO * duration), endpoint=False)
    left_orig = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right_orig = 0.5 * np.sin(2 * np.pi * 1500 * t)
    
    left_pre = apply_preemphasis(left_orig, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    right_pre = apply_preemphasis(right_orig, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    composite, _, _, _, _, _ = stereo_multiplex(
        left_pre, right_pre, FMStereoParams.FS_AUDIO, FMStereoParams.FS_COMPOSITE
    )
    
    fm_clean, _, _, _ = fm_modulate(
        composite, FMStereoParams.FS_COMPOSITE, FMStereoParams.FS_FM,
        FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
    )
    
    for snr_in in input_snrs:
        print(f"\n--- Testing Input SNR = {snr_in} dB ---")
        
        fm_noisy = add_awgn(fm_clean, snr_in)
        
        composite_rec = fm_demodulate(
            fm_noisy, FMStereoParams.FS_FM, FMStereoParams.FS_COMPOSITE,
            FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
        )
        
        left_rec, right_rec, _, _ = stereo_decode(composite_rec, FMStereoParams.FS_COMPOSITE)
        
        left_rec = np.nan_to_num(left_rec, nan=0.0)
        right_rec = np.nan_to_num(right_rec, nan=0.0)
        
        snr_out = calculate_snr(left_orig, left_rec)
        
        results.append({
            'input_snr': snr_in,
            'output_snr': snr_out,
            'left_rec': left_rec,
            'right_rec': right_rec
        })
        
        print(f"  Output SNR: {snr_out:.1f} dB")
    
    # Separation test
    left_only = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right_zero = np.zeros_like(t)
    left_only_pre = apply_preemphasis(left_only, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    right_zero_pre = apply_preemphasis(right_zero, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    comp_sep, _, _, _, _, _ = stereo_multiplex(
        left_only_pre, right_zero_pre, FMStereoParams.FS_AUDIO, FMStereoParams.FS_COMPOSITE
    )
    fm_sep, _, _, _ = fm_modulate(
        comp_sep, FMStereoParams.FS_COMPOSITE, FMStereoParams.FS_FM,
        FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
    )
    
    print("\n--- Channel Separation vs Input SNR ---")
    for snr_in in input_snrs:
        fm_sep_noisy = add_awgn(fm_sep, snr_in)
        comp_sep_rec = fm_demodulate(
            fm_sep_noisy, FMStereoParams.FS_FM, FMStereoParams.FS_COMPOSITE,
            FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
        )
        left_sep_rec, right_sep_rec, _, _ = stereo_decode(comp_sep_rec, FMStereoParams.FS_COMPOSITE)
        sep = measure_separation(left_sep_rec, right_sep_rec)
        
        for r in results:
            if r['input_snr'] == snr_in:
                r['separation'] = sep
        print(f"  SNR {snr_in} dB: Separation = {sep:.1f} dB")
    
    # Plot: SNR Transfer
    fig, ax = plt.subplots(figsize=(10, 6))
    in_snrs = [r['input_snr'] for r in results]
    out_snrs = [r['output_snr'] for r in results]
    
    ax.plot(in_snrs, out_snrs, 'bo-', markersize=10, linewidth=2, label='Measured')
    ax.plot(in_snrs, in_snrs, 'r--', linewidth=1, alpha=0.7, label='Unity (no improvement)')
    ax.set_xlabel('Input SNR (dB)', fontsize=12)
    ax.set_ylabel('Output SNR (dB)', fontsize=12)
    ax.set_title('Task 2a: FM SNR Transfer Characteristic', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for x, y in zip(in_snrs, out_snrs):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                   xytext=(0, 8), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task2a_snr_transfer.png'), dpi=150)
    plt.close()
    
    # Plot: Separation vs Noise
    fig, ax = plt.subplots(figsize=(10, 6))
    seps = [r.get('separation', 0) for r in results]
    
    ax.bar(in_snrs, seps, color='steelblue', width=3)
    ax.set_xlabel('Input SNR (dB)', fontsize=12)
    ax.set_ylabel('Channel Separation (dB)', fontsize=12)
    ax.set_title('Task 2b: Channel Separation vs Input SNR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20 dB threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task2b_separation_vs_noise.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")
    return results

if __name__ == "__main__":
    run_noise_immunity_analysis()
