import os
import numpy as np
import random
from data_loader import load_all_records
from preprocessing import preprocess_all 

SPLITS_DIR = './data/splits'
RANDOM_SEED = 42
TEST_SIZE = 0.25

def make_splits(processed):
    
    all_records = list(processed.keys())
    
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)
    
    split_idx = int(len(all_records) * (1 - TEST_SIZE))
    train_records = all_records[:split_idx]
    test_records = all_records[split_idx:]

    train_beats = []
    test_beats = []
    test_labels = []
    
    for rid, (beats, labels) in processed.items():
        
        if rid in train_records:
            
            normal_mask = labels == 0
            train_beats.append(beats[normal_mask])
        
        elif rid in test_records:
            
            test_beats.append(beats)
            test_labels.append(labels)
    
    train = np.vstack(train_beats)
    test_X = np.vstack(test_beats)
    test_y = np.concatenate(test_labels)

    os.makedirs(SPLITS_DIR, exist_ok=True)
    np.save(f'{SPLITS_DIR}/train_normal.npy', train)
    np.save(f'{SPLITS_DIR}/test_beats.npy', test_X)
    np.save(f'{SPLITS_DIR}/test_labels.npy', test_y)
    
    print(f"\nTotal patients: {len(all_records)}")
    print(f"Train patients: {len(train_records)} — {train_records}")
    print(f"Test patients: {len(test_records)}  — {test_records}")
    print(f"Train beats: {train.shape}")
    print(f"Test beats: {test_X.shape}")
    print(f"Test normal: {(test_y==0).sum():,} ({(test_y==0).mean()*100:.1f}%)")
    print(f"Test anomaly: {(test_y==1).sum():,} ({(test_y==1).mean()*100:.1f}%)")
    print(f"\nSaved to {SPLITS_DIR}/")
    
    return train, test_X, test_y


if __name__ == "__main__":
    
    if (os.path.exists(f'{SPLITS_DIR}/train_normal.npy') and
        os.path.exists(f'{SPLITS_DIR}/test_beats.npy') and
        os.path.exists(f'{SPLITS_DIR}/test_labels.npy')):
        print("Splits already exist in data folder!!!")
    
    else:
        data = load_all_records()
        processed = preprocess_all(data)
        make_splits(processed)
