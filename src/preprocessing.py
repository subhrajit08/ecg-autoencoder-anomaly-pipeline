import numpy as np
from scipy.signal import butter, filtfilt
from data_loader import load_all_records

FS = 360
WINDOW = 90
NORMAL_BEATS  = {'N', 'L', 'R', 'e', 'j'}
ANOMALY_BEATS = {'V', 'A', 'F', '/', 'f', 'E', 'J', 'a', 'S'}
SKIP_BEATS    = {'~', '|', '+', '[', ']', '!', 'x', 'Q'}

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS):
    
    nyq  = fs / 2.0
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    
    return filtfilt(b, a, signal, axis=0) 

def segment_beats(signal, r_peaks, symbols, window=WINDOW):
    
    beats, labels = [], []
    sig = signal[:, 0] 
    
    for peak, sym in zip(r_peaks, symbols):
        
        if sym in SKIP_BEATS:
            continue
        
        start, end = peak - window, peak + window
        if start < 0 or end > len(sig):
            continue
        
        if sym not in NORMAL_BEATS and sym not in ANOMALY_BEATS:
            continue
        
        beat  = sig[start:end]
        label = 0 if sym in NORMAL_BEATS else 1

        beats.append(beat)
        labels.append(label)

    return  np.array(beats,  dtype=np.float32), \
            np.array(labels, dtype=np.int8)

def normalize_beats(beats):
    
    mean = beats.mean(axis=1, keepdims=True)
    std  = beats.std(axis=1, keepdims=True) + 1e-8
    
    return (beats - mean) / std

def preprocess_all(data):
    
    processed = {}
    
    for rid, (signal, r_peaks, symbols) in data.items():
        
        filtered = bandpass_filter(signal)
        beats, labels = segment_beats(filtered, r_peaks, symbols)
        beats = normalize_beats(beats)
        processed[rid] = (beats, labels)
        
        print(
            f"Record {rid:>3} | "
            f"Beats: {len(beats):>5} | "
            f"Normal: {(labels==0).sum():>5} | "
            f"Anomaly: {(labels==1).sum():>4}"
        )
        
    return processed


if __name__ == "__main__":

    data = load_all_records()
    processed = preprocess_all(data)

    sample_rid = list(processed.keys())[0]
    beats, labels = processed[sample_rid]
    print(f"\nSample record: {sample_rid}")
    print(f"Beats shape: {beats.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Beat min/max: {beats.min():.4f} / {beats.max():.4f}")
    print(f"Normal beats: {(labels==0).sum()}")
    print(f"Anomaly beats: {(labels==1).sum()}")
