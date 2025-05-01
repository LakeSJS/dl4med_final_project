import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Helper to safely convert strings to floats
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# Mapping sleep-stage labels to integers
SLEEP_STAGE_MAPPING = {
    "W": 0,    # Wake
    "N1": 1,   # non-REM stage 1
    "N2": 1,   # non-REM stage 2 also light sleep
    "N3": 2,   # non-REM stage 3
    "R": 3,    # REM
    "Missing": -1  # Missing label → ignore
}

# Forward‑fill NaNs in each channel
def forward_fill(x: torch.Tensor) -> torch.Tensor:
    single = False
    if x.dim() == 1:
        x = x.unsqueeze(1)
        single = True
    T, C = x.shape
    for c in range(C):
        if torch.isnan(x[0, c]):
            x[0, c] = 0.0
        for t in range(1, T):
            if torch.isnan(x[t, c]):
                x[t, c] = x[t - 1, c]
    return x.squeeze(1) if single else x

# Numeric columns for the CSV reader
numeric_columns = ['TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR', 'IBI']
converters = {col: safe_float for col in numeric_columns}


class Wearables(Dataset):
    def __init__(self,
                 subjects_list,
                 data_dir,
                 chunk_duration: float = 600,
                 chunk_stride: float = 300,
                 downsample_freq: int = 64,
                 acc_freq: int = 64,
                 feature_columns=None,
                 debug: bool = False):
        """
        Returns for each chunk:
          - non-acceleration features at `downsample_freq`
          - acceleration at `acc_freq`
          - labels at `downsample_freq`
        """
        # choose features (ACC will be computed)
        if feature_columns is None:
            self.feature_columns = ['ACC_X', 'ACC_Y', 'ACC_Z', 'TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI']
        else:
            self.feature_columns = feature_columns

        # factors and lengths
        self.downsample     = int(64 // downsample_freq)
        self.downsample_acc = int(64 // acc_freq)
        self.chunk_length      = int(chunk_duration * downsample_freq)
        self.chunk_length_acc  = int(chunk_duration * acc_freq)
        self.stride            = int(chunk_stride * downsample_freq)
        # to align non‐acc → acc indices
        self.ratio = acc_freq / downsample_freq
        # which columns to keep *besides* ACC
        self.non_acc_idxs = [
            i for i, c in enumerate(self.feature_columns)
            if c not in ['ACC_X', 'ACC_Y', 'ACC_Z']
        ]

        self.chunks = []
        for SID in subjects_list:
            path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if not os.path.exists(path):
                if debug:
                    print(f"[WARN] Missing file for {SID}, skipping")
                continue

            # 1) load
            df = pd.read_csv(path,
                             dtype={'Sleep_Stage': 'category'},
                             converters=converters,
                             low_memory=True)
            if debug:
                print(f"[INFO] {SID}: {len(df)} rows loaded")

            if self.downsample_acc != 1:
                df = df.iloc[::self.downsample_acc].reset_index(drop=True)
                if debug:
                    print(f"[DEBUG] {SID}: ACC ↓ to {int(64/self.downsample_acc)} Hz → {len(df)} rows")
            df_acc = (df[['ACC_X', 'ACC_Y', 'ACC_Z']] - df[['ACC_X', 'ACC_Y', 'ACC_Z']].mean()) / df[['ACC_X', 'ACC_Y', 'ACC_Z']].std()
            acc_arr = df_acc.to_numpy(dtype=np.float32)

            # 3) downsample *all* channels for non‐acc view
            df = df.iloc[::self.downsample].reset_index(drop=True)
            if debug:
                print(f"[DEBUG] {SID}: non-ACC ↓ to {int(64/self.downsample)} Hz → {len(df)} rows")

            # 4) drop preparation phase, map labels
            df = df[df['Sleep_Stage'] != 'P']
            df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
            labels_arr = (
                df['Sleep_Stage']
                  .map(SLEEP_STAGE_MAPPING)
                  .fillna(-1)
                  .astype(int)
                  .to_numpy()
            )
            # 5) normalize features
            for c in self.feature_columns:
                df[c] = (df[c] - df[c].mean()) / df[c].std() # normalizing per subject, should we do globally?

            # 6) assemble feature matrix
            data_arr = df[self.feature_columns].values.astype(np.float32)

            # 7) pad short records
            T = data_arr.shape[0]
            if T < self.chunk_length:
                pad = self.chunk_length - T
                data_arr   = np.vstack([data_arr,
                                        np.full((pad, data_arr.shape[1]), np.nan,
                                                dtype=np.float32)])
                labels_arr = np.concatenate(
                    [labels_arr, np.full((pad,), -1, dtype=int)]
                )
                T = self.chunk_length

            # 8) slice into overlapping chunks
            for start in range(0, T - self.chunk_length + 1, self.stride):
                end       = start + self.chunk_length
                start_acc = int(start * self.ratio)
                end_acc   = start_acc + self.chunk_length_acc

                non_acc_chunk = forward_fill(torch.tensor(data_arr[start:end, self.non_acc_idxs]))
                acc_chunk     = forward_fill(torch.tensor(acc_arr[start_acc:end_acc, :]))
                label_chunk   = labels_arr[start:end]

                self.chunks.append({
                    'non_acc': non_acc_chunk,
                    'acc':     acc_chunk,
                    'labels':  label_chunk
                })

        if debug:
            print(f"[INFO] Built {len(self.chunks)} total chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        non_acc = c['non_acc']
        acc     = c['acc']
        labels  = c['labels']

        # forward‑fill each
        non_acc = non_acc
        #print(acc.shape, acc.unsqueeze(1).shape)
        acc     = acc

        return non_acc, acc, labels