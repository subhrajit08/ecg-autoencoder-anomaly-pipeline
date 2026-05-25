import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

FS = 360
OUTPUT_DIR = './outputs'


def compute_rr_intervals(r_peaks):
    rr_samples = np.diff(r_peaks)
    rr_seconds = rr_samples / FS
    rr_ms = rr_seconds * 1000
    return rr_ms


def compute_heart_rate(rr_ms):
    mean_rr_seconds = rr_ms.mean() / 1000
    hr = 60 / mean_rr_seconds
    return round(hr, 2)


def compute_hrv(rr_ms):
    return round(float(np.std(rr_ms)), 2)


def compute_nn50(rr_ms):
    successive_diff = np.abs(np.diff(rr_ms))
    return int(np.sum(successive_diff > 50))


def compute_r_peak_amplitudes(signal, r_peaks):
    sig = signal[:, 0]
    amplitudes = sig[r_peaks]
    return amplitudes


def classify_rhythm(rr_ms, nn50):
    cv = np.std(rr_ms) / np.mean(rr_ms)
    if nn50 > 50 and cv > 0.15:
        return "Atrial Fibrillation"
    elif cv < 0.10:
        return "Normal Sinus Rhythm"
    else:
        return "Other/Irregular Rhythm"


def classify_hr_condition(hr):
    if hr > 100:
        return "Tachycardia (Rapid)"
    elif hr < 60:
        return "Bradycardia (Slow)"
    else:
        return "Normal"


def generate_suggestion(rhythm, hr_condition):
    if rhythm == "Atrial Fibrillation" and hr_condition == "Tachycardia (Rapid)":
        return "URGENT: Atrial fibrillation with rapid ventricular response detected. Seek immediate medical attention."
    elif rhythm == "Atrial Fibrillation":
        return "WARNING: Atrial fibrillation detected. Consult a cardiologist promptly."
    elif hr_condition == "Tachycardia (Rapid)":
        return "CAUTION: Resting heart rate is elevated. Monitor symptoms and consult a physician if persistent."
    elif hr_condition == "Bradycardia (Slow)":
        return "CAUTION: Heart rate is below normal. Consult a physician if experiencing dizziness or fatigue."
    elif rhythm == "Other/Irregular Rhythm":
        return "NOTICE: Irregular rhythm detected. A physician review is recommended."
    else:
        return "NORMAL: No significant abnormalities detected. Continue routine monitoring."


def analyse_record(signal, r_peaks, record_id=None):
    rr_ms = compute_rr_intervals(r_peaks)
    hr = compute_heart_rate(rr_ms)
    hrv = compute_hrv(rr_ms)
    nn50 = compute_nn50(rr_ms)
    avg_rr = round(float(rr_ms.mean()), 2)
    amplitudes = compute_r_peak_amplitudes(signal, r_peaks)
    rhythm = classify_rhythm(rr_ms, nn50)
    hr_condition = classify_hr_condition(hr)
    suggestion = generate_suggestion(rhythm, hr_condition)

    results = {
        'record_id': record_id,
        'total_beats': len(r_peaks),
        'heart_rate': hr,
        'avg_rr_ms': avg_rr,
        'hrv_sdnn' : hrv,
        'nn50': nn50,
        'r_peak_amps': amplitudes,
        'rhythm' : rhythm,
        'hr_condition': hr_condition,
        'suggestion': suggestion,
    }

    return results


def print_report(results):
    print("=" * 50)
    print(f"ECG ANALYSIS REPORT — Record {results['record_id']}")
    print("=" * 50)
    print(f"Total beats: {results['total_beats']}")
    print(f"Heart rate: {results['heart_rate']} bpm")
    print(f"Avg RR interval: {results['avg_rr_ms']} ms")
    print(f"HRV (SDNN): {results['hrv_sdnn']} ms")
    print(f"NN50: {results['nn50']}")
    print(f"R-peak amp range: {results['r_peak_amps'].min():.3f} — {results['r_peak_amps'].max():.3f} mV")
    print(f"Rhythm: {results['rhythm']}")
    print(f"HR condition: {results['hr_condition']}")
    print(f"Suggestion: {results['suggestion']}")
    print("=" * 50)


def plot_analysis(signal, r_peaks, results, record_id=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sig = signal[:, 0]
    rr_ms = compute_rr_intervals(r_peaks)

    start = r_peaks[5] - 200 if len(r_peaks) > 5 else 0
    end = r_peaks[20] + 200 if len(r_peaks) > 20 else len(sig)
    segment = sig[start:end]
    r_in_seg = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
    time_axis = np.arange(len(segment)) / FS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle(f'ECG Analysis Report — Record {record_id}', color='white', fontsize=13)
    colors = {'signal': '#00d4aa', 'rpeaks': '#ff4d6d', 'rr': '#f4a261', 'amp': '#a78bfa'}

    ax = axes[0][0]
    ax.set_facecolor('#1a1a2e')
    ax.plot(time_axis, segment, color=colors['signal'], linewidth=0.8)
    ax.scatter(r_in_seg / FS, segment[r_in_seg], color=colors['rpeaks'], s=30, zorder=5, label='R-peaks')
    ax.set_title('ECG Signal with R-peaks', color='white')
    ax.set_xlabel('Time (s)', color='gray')
    ax.set_ylabel('Amplitude (mV)', color='gray')
    ax.tick_params(colors='gray')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    ax = axes[0][1]
    ax.set_facecolor('#1a1a2e')
    ax.plot(rr_ms, color=colors['rr'], linewidth=1.2)
    ax.axhline(y=rr_ms.mean(), color='white', linestyle='--', linewidth=0.8, label=f'Mean: {rr_ms.mean():.1f} ms')
    ax.set_title('RR Intervals', color='white')
    ax.set_xlabel('Beat index', color='gray')
    ax.set_ylabel('RR interval (ms)', color='gray')
    ax.tick_params(colors='gray')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    ax = axes[1][0]
    ax.set_facecolor('#1a1a2e')
    ax.plot(results['r_peak_amps'], color=colors['amp'], linewidth=1.0)
    ax.axhline(y=results['r_peak_amps'].mean(), color='white', linestyle='--',
            linewidth=0.8, label=f'Mean: {results["r_peak_amps"].mean():.3f} mV')
    ax.set_title('R-peak Amplitudes', color='white')
    ax.set_xlabel('Beat index', color='gray')
    ax.set_ylabel('Amplitude (mV)', color='gray')
    ax.tick_params(colors='gray')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    ax = axes[1][1]
    ax.set_facecolor('#1a1a2e')
    ax.hist(rr_ms, bins=40, color=colors['rr'], alpha=0.8, edgecolor='none')
    ax.set_title('RR Interval Distribution (HRV)', color='white')
    ax.set_xlabel('RR interval (ms)', color='gray')
    ax.set_ylabel('Count', color='gray')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

    summary = (
        f"HR: {results['heart_rate']} bpm   "
        f"Avg RR: {results['avg_rr_ms']} ms   "
        f"HRV: {results['hrv_sdnn']} ms   "
        f"NN50: {results['nn50']}\n"
        f"Rhythm: {results['rhythm']}   "
        f"Condition: {results['hr_condition']}\n"
        f"{results['suggestion']}"
    )
    fig.text(0.5, 0.01, summary, ha='center', color='white', fontsize=9,
            family='monospace', bbox=dict(facecolor='#1a1a2e', edgecolor='#333', boxstyle='round'))

    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    path = f'{OUTPUT_DIR}/ecg_analysis_{record_id}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    from data_loader import load_all_records

    data = load_all_records()
    rid = list(data.keys())[0]
    signal, r_peaks, symbols = data[rid]

    results = analyse_record(signal, r_peaks, record_id=rid)
    print_report(results)
    plot_analysis(signal, r_peaks, results, record_id=rid)