"""
Analysis Task 5: System Robustness Test
Tests pilot frequency errors from -500 Hz to +500 Hz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fm_stereo_system import (
    FMStereoParams, apply_preemphasis, apply_deemphasis,
    fm_modulate, fm_demodulate
)
import os

def stereo_multiplex_offset_pilot(left, right, fs_audio, fs_composite, pilot_offset=0):
    """Create stereo composite with offset pilot frequency."""
    upsample = fs_composite / fs_audio
    num_samples = int(len(left) * upsample)
    
    left_up = signal.resample(left, num_samples)
    right_up = signal.resample(right, num_samples)
    t = np.arange(num_samples) / fs_composite
    
    L_plus_R = (left_up + right_up) / 2
    L_minus_R = (left_up - right_up) / 2
    
    pilot_freq = 19000 + pilot_offset
    pilot = 0.1 * np.sin(2 * np.pi * pilot_freq * t)
    subcarrier = np.sin(2 * np.pi * 2 * pilot_freq * t)
    L_minus_R_mod = L_minus_R * subcarrier
    
    composite = L_plus_R + pilot + L_minus_R_mod
    composite = composite / np.max(np.abs(composite))
    
    return composite, pilot_freq

def stereo_decode_fixed_freq(composite, fs):
    """Stereo decoder expecting exactly 19 kHz pilot."""
    t = np.arange(len(composite)) / fs
    nyq = fs / 2
    
    sos_lp = signal.butter(5, 15000/nyq, btype='low', output='sos')
    L_plus_R = signal.sosfiltfilt(sos_lp, composite)
    
    sos_pilot = signal.butter(4, [(19000-150)/nyq, (19000+150)/nyq], btype='band', output='sos')
    pilot_ext = signal.sosfiltfilt(sos_pilot, composite)
    
    pilot_power = np.sqrt(np.mean(pilot_ext**2))
    pilot_max = np.max(np.abs(pilot_ext))
    
    if pilot_max < 1e-10:
        carrier_38 = np.sin(2 * np.pi * 38000 * t)
    else:
        pilot_norm = pilot_ext / pilot_max
        analytic = signal.hilbert(pilot_norm)
        carrier_raw = -np.imag(analytic ** 2)
        
        sos_38 = signal.butter(4, [(38000-500)/nyq, (38000+500)/nyq], btype='band', output='sos')
        carrier_38 = signal.sosfiltfilt(sos_38, carrier_raw)
        c_max = np.max(np.abs(carrier_38))
        carrier_38 = carrier_38 / c_max if c_max > 1e-10 else np.sin(2 * np.pi * 38000 * t)
    
    sos_lr = signal.butter(4, [23000/nyq, min(53000/nyq, 0.95)], btype='band', output='sos')
    L_minus_R_mod = signal.sosfiltfilt(sos_lr, composite)
    L_minus_R = signal.sosfiltfilt(sos_lp, L_minus_R_mod * carrier_38 * 2)
    
    left = L_plus_R + L_minus_R
    right = L_plus_R - L_minus_R
    
    n = int(len(left) * FMStereoParams.FS_AUDIO / fs)
    left_out = apply_deemphasis(signal.resample(left, n), FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    right_out = apply_deemphasis(signal.resample(right, n), FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    return np.nan_to_num(left_out), np.nan_to_num(right_out), pilot_power

def run_robustness_analysis():
    """Test robustness to pilot frequency errors."""
    output_dir = "output/task5_robustness"
    os.makedirs(output_dir, exist_ok=True)
    
    offsets = np.arange(-500, 501, 50)
    results = []
    
    print("=" * 60)
    print("TASK 5: SYSTEM ROBUSTNESS TEST")
    print("=" * 60)
    
    t = np.linspace(0, 1, FMStereoParams.FS_AUDIO, endpoint=False)
    left_orig = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right_orig = np.zeros_like(t)
    
    left_pre = apply_preemphasis(left_orig, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    right_pre = apply_preemphasis(right_orig, FMStereoParams.TAU, FMStereoParams.FS_AUDIO)
    
    for offset in offsets:
        composite, _ = stereo_multiplex_offset_pilot(
            left_pre, right_pre, FMStereoParams.FS_AUDIO,
            FMStereoParams.FS_COMPOSITE, pilot_offset=offset
        )
        
        fm_sig, _, _, _ = fm_modulate(
            composite, FMStereoParams.FS_COMPOSITE, FMStereoParams.FS_FM,
            FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
        )
        
        comp_rec = fm_demodulate(
            fm_sig, FMStereoParams.FS_FM, FMStereoParams.FS_COMPOSITE,
            FMStereoParams.FREQ_DEVIATION, FMStereoParams.FC
        )
        
        left_rec, right_rec, pilot_power = stereo_decode_fixed_freq(comp_rec, FMStereoParams.FS_COMPOSITE)
        
        l_rms = np.sqrt(np.mean(left_rec**2))
        r_rms = np.sqrt(np.mean(right_rec**2))
        sep = np.clip(20 * np.log10(l_rms / r_rms) if r_rms > 1e-12 else 60, -10, 60)
        
        results.append({'offset': offset, 'separation': sep, 'pilot_power': pilot_power})
        
        if offset in [-500, -250, 0, 250, 500]:
            print(f"  Offset {offset:+4d} Hz: Separation = {sep:.1f} dB")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 5: System Robustness - Pilot Frequency Error', fontsize=14, fontweight='bold')
    
    offs = [r['offset'] for r in results]
    seps = [r['separation'] for r in results]
    powers = [r['pilot_power'] for r in results]
    
    axes[0, 0].plot(offs, seps, 'b-o', markersize=4, linewidth=1.5)
    axes[0, 0].axhline(y=15, color='r', linestyle='--', linewidth=2, label='15 dB threshold')
    axes[0, 0].set_xlabel('Pilot Frequency Offset (Hz)')
    axes[0, 0].set_ylabel('Channel Separation (dB)')
    axes[0, 0].set_title('Channel Separation vs Pilot Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([-550, 550])
    
    axes[0, 1].plot(offs, powers, 'g-o', markersize=4, linewidth=1.5)
    axes[0, 1].set_xlabel('Pilot Frequency Offset (Hz)')
    axes[0, 1].set_ylabel('Extracted Pilot Power')
    axes[0, 1].set_title('Pilot Extraction Strength vs Offset')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Find threshold
    threshold = max([abs(r['offset']) for r in results if r['separation'] >= 15], default=0)
    
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    sep_0 = next((r['separation'] for r in results if r['offset'] == 0), 0)
    sep_500 = next((r['separation'] for r in results if r['offset'] == 500), 0)
    
    summary = f"""
    ROBUSTNESS ANALYSIS
    ===================
    
    Separation at 0 Hz offset: {sep_0:.1f} dB
    Separation at ±500 Hz: {sep_500:.1f} dB
    
    Tolerance for ≥15 dB: ±{threshold} Hz
    
    Finding:
    Receiver's pilot extraction filter
    cannot track offset pilot, causing
    carrier regeneration failure.
    """
    axes[1, 0].text(0.1, 0.9, summary, transform=axes[1, 0].transAxes,
                    fontsize=12, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task5_robustness.png'), dpi=150)
    plt.close()
    
    print(f"\nMax error for ≥15 dB: ±{threshold} Hz")
    print(f"Plot saved to {output_dir}")
    return results

if __name__ == "__main__":
    run_robustness_analysis()
