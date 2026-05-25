import os
import wfdb
import time

RAW_DIR = './data/raw/mitdb'

RECORDS = [int(r_id) for r_id in wfdb.get_record_list('mitdb')]

def download_dataset():
    
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading {len(RECORDS)} records...")
    
    for r_id in RECORDS:
        save_path = os.path.join(RAW_DIR, str(r_id))
        if os.path.exists(save_path + '.hea'):
            print(f"Record {r_id} already exists, skipping...")
            continue
        for attempt in range(5):
            try:
                wfdb.dl_database('mitdb', dl_dir=RAW_DIR, records=[str(r_id)])
                print(f"Downloaded record {r_id}")
                break
            except Exception as e:
                print(f"Record {r_id} attempt {attempt+1} failed: {e}")
                time.sleep(10)
    
    print("Download complete.")

def load_record(record_id):
    
    path = f"{RAW_DIR}/{record_id}"
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')
    signal = record.p_signal
    r_peaks = annotation.sample
    symbols = annotation.symbol
    
    return signal, r_peaks, symbols

def load_all_records():
    
    data = {}
    
    for r_id in RECORDS:
        signal, r_peaks, symbols = load_record(r_id)
        data[r_id] = (signal, r_peaks, symbols)
        print(f"Loaded Record {r_id}")
    
    return data


if __name__ == "__main__":
    
    download_dataset()
    data = load_all_records()
    print(f"\nRecords: {RECORDS}")
    print(f"\nTotal records loaded: {len(data)}")
