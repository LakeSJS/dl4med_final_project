#!/usr/bin/env python
# coding: utf-8

# # Investigation of the impact of causality constraints on a CNN-LSTM sleep staging model
# 
# ### Project goals:
# ##### Classify sleep stages using multi-modal sensor data (BVP, accelerometer, timestamps, temperature).
# ##### Compare model performance and computation between non-causal versus causal architectures.

# ## Library Imports

# In[ ]:


import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassCohenKappa
from IPython.display import clear_output
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from x_transformers import ContinuousTransformerWrapper, Encoder


# In[ ]:


# demo csv, get columns
demo_csv_path = '/gpfs/data/oermannlab/users/slj9342/dl4med_25/data/physionet.org/files/dreamt/2.0.0/data_64Hz/S016_whole_df.csv'
demo_df = pd.read_csv(demo_csv_path)
print(demo_df.columns)
col_dtypes = demo_df.dtypes
print(col_dtypes)


# ## Data loading

# In[ ]:


datadir_64Hz = '/gpfs/data/oermannlab/users/slj9342/dl4med_25/data/physionet.org/files/dreamt/2.0.0/data_64Hz/' # working with 64Hz data

dtype_dict = {
    'TIMESTAMP': np.float32,
    'BVP': np.float32,
    'ACC_X': np.float32,
    'ACC_Y': np.float32,
    'ACC_Z': np.float32,
    'TEMP': np.float32,
    'EDA': np.float32,
    'HR': np.float32,
    'IBI': np.float32,
    'Sleep_Stage': 'category',
    'Obstructive_Apnea': 'Int64', 
    'Central_Apnea': 'Int64',
    'Hypopnea': 'Int64',
    'Multiple_Events': 'Int64'
}

file = 'S016_whole_df.csv'
df_head = pd.read_csv(os.path.join(datadir_64Hz, file), nrows=5)
print(df_head.columns.tolist())

df = pd.read_csv(os.path.join(datadir_64Hz, file))
print(df.dtypes)

df = pd.read_csv(os.path.join(datadir_64Hz, file), low_memory=False)
for col in df.columns:
    try:
        pd.to_numeric(df[col], errors='raise')
    except Exception as e:
        print(f"❌ Column {col} failed: {e}")



# In[ ]:


# get max sequence length
def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

numeric_columns = [
    'TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP',
    'EDA', 'HR', 'IBI'
]
converters = {col: safe_float for col in numeric_columns}
'''
max_length = 0
for file in os.listdir(datadir_64Hz):
    if file.endswith('_whole_df.csv'):
        df = pd.read_csv(
            os.path.join(datadir_64Hz, file),
            dtype={'Sleep_Stage': 'category'},
            converters=converters,
            low_memory=True
        )
        max_length = max(max_length, len(df))
print(f"Max sequence length: {max_length}")
'''
max_length = 2493810


# ### Split subjects into train, val, and test

# In[ ]:


participant_info_df = pd.read_csv('/gpfs/data/oermannlab/users/slj9342/dl4med_25/data/physionet.org/files/dreamt/2.0.0/participant_info.csv')
subjects_all = participant_info_df['SID']

subjects_all_shuffled = participant_info_df['SID'].sample(frac=1, random_state=42).reset_index(drop=True)
subjects_train = subjects_all_shuffled[:int(len(subjects_all_shuffled)*0.8)]
subjects_val = subjects_all_shuffled[int(len(subjects_all_shuffled)*0.8):int(len(subjects_all_shuffled)*0.9)]
subjects_test = subjects_all_shuffled[int(len(subjects_all_shuffled)*0.9):]
print(f"number of subjects in train: {len(subjects_train)}")
print(f"number of subjects in val: {len(subjects_val)}")
print(f"number of subjects in test: {len(subjects_test)}")

# overwrite with smaller dataset for development (20% of original)
fraction = 0.3
subjects_train_small = subjects_train[:int(len(subjects_train)*fraction)]
subjects_val_small = subjects_val[:int(len(subjects_val)*fraction)]
subjects_test_small = subjects_test[:int(len(subjects_test)*fraction)]
print(f"number of subjects in small train: {len(subjects_train_small)}")
print(f"number of subjects in small val: {len(subjects_val_small)}")
print(f"number of subjects in small test: {len(subjects_test_small)}")


# ### Non-windowed dataset class
# 
# 
# 

# In[ ]:


SLEEP_STAGE_MAPPING = {
    "W": 0,    # Wake
    "N1": 1,   # non-REM stage 1
    "N2": 2,   # non-REM stage 2
    "N3": 3,   # non-REM stage 3
    "R": 4,    # REM
    "Missing": -1  # Missing label
}

def forward_fill(x):
    """
    Performs forward fill on a tensor. If x is 1D (shape [T]),
    it's temporarily unsqueezed to [T, 1] and then processed.
    Assumes the first value is valid, or fills it with zero if needed.
    """
    single_channel = False
    if x.dim() == 1:
        x = x.unsqueeze(1)
        single_channel = True
    
    T, C = x.shape
    for c in range(C):
        # Optionally, handle the first element if it's NaN
        if torch.isnan(x[0, c]):
            x[0, c] = 0.0  # or choose another default value
        for t in range(1, T):
            if torch.isnan(x[t, c]):
                x[t, c] = x[t - 1, c]
    
    if single_channel:
        x = x.squeeze(1)
    return x

class SleepDataset(Dataset):
    def __init__(self, subjects_list, data_dir, max_length, downsample_freq=64, debug=False):
        self.subjects = [{} for _ in range(len(subjects_list))]
        self.downsample = int(64 // downsample_freq)  # Downsample factor
        self.max_length = int(max_length // self.downsample)

        for subjectNo, SID in enumerate(subjects_list):
            # Load the data for each subject
            file_path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(
                    file_path,
                    dtype={'Sleep_Stage': 'category'},
                    converters=converters,
                    low_memory=True
                )
                if debug:
                    print(f"loaded data for {SID}:")

                # Downsample the data if needed
                if self.downsample != 1:
                    df = df.iloc[::self.downsample].reset_index(drop=True)
                    if debug:
                        print(f"After downsampling by factor {self.downsample}, rows: {len(df)}")
                
                df = df[df['Sleep_Stage'] != 'P'] # remove data before PSG start
                for col in ['ACC_X', 'ACC_Y', 'ACC_Z','BVP', 'TEMP', 'TIMESTAMP']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                ACC = np.sqrt(df['ACC_X']**2 + df['ACC_Y']**2 + df['ACC_Z']**2) # assuming its unlikely each acc channel really carries important information
                df_X = df[['TIMESTAMP', 'BVP', 'TEMP']].copy()
                df_X['ACC'] = ACC
                # Normalize the features (z-score normalization per subject)
                TEMP_norm = (df_X['TEMP'] - df_X['TEMP'].mean()) / df_X['TEMP'].std()
                df_X['TEMP'] = TEMP_norm
                BVP_norm = (df_X['BVP'] - df_X['BVP'].mean()) / df_X['BVP'].std()
                df_X['BVP'] = BVP_norm
                df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
                df_Y = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING)
                
                # Pad/truncate the data to the downsampled max_length
                if len(df_X) > self.max_length:
                    if debug:
                        print(f"Truncating data for {SID} from {len(df_X)} to {self.max_length} samples.")
                    df_X = df_X.iloc[:self.max_length]
                    df_Y = df_Y.iloc[:self.max_length]
                else:
                    padding_length = self.max_length - len(df_X)
                    padding = pd.DataFrame(np.nan, index=np.arange(padding_length), columns=df_X.columns)
                    df_X = pd.concat([df_X, padding], ignore_index=True)
                    df_Y = pd.concat([df_Y, pd.Series([-1] * padding_length)], ignore_index=True)
                self.subjects[subjectNo] = {
                    'data': df_X.values.astype(np.float32),  # shape: [T, C]
                    'labels': df_Y.to_numpy(),                 # shape: [T]
                    'SID': SID
                }
                if debug:
                    print(f"Data shape for {SID}: {df_X.shape}, Labels shape: {df_Y.shape}")
            else:
                warning(f"File {file_path} does not exist. Skipping subject {SID}.")
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        data = torch.tensor(subject['data'], dtype=torch.float32)
        labels = torch.tensor(subject['labels'], dtype=torch.long)

        data = forward_fill(data) # fill NaNs with previous values
        return data, labels


# ### Mixed Frequency Dataset

# In[ ]:





# ### Chunked Dataset Class

# In[ ]:


# Sleep stage mapping as before
SLEEP_STAGE_MAPPING = {
    "W": 0,    # Wake
    "N1": 1,   # non-REM stage 1
    "N2": 2,   # non-REM stage 2
    "N3": 3,   # non-REM stage 3
    "R": 4,    # REM
    "Missing": -1  # Missing label
}

def forward_fill(x):
    """
    Performs forward fill on a tensor.
    If x is 1D (shape [T]), it is temporarily unsqueezed to [T, 1].
    Assumes the first value is valid, or fills it with zero if needed.
    """
    single_channel = False
    if x.dim() == 1:
        x = x.unsqueeze(1)
        single_channel = True

    T, C = x.shape
    for c in range(C):
        if torch.isnan(x[0, c]):
            x[0, c] = 0.0
        for t in range(1, T):
            if torch.isnan(x[t, c]):
                x[t, c] = x[t - 1, c]
    if single_channel:
        x = x.squeeze(1)
    return x

numeric_columns = [
    'TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP',
    'EDA', 'HR', 'IBI'
]
converters = {col: safe_float for col in numeric_columns}

class SleepChunkDataset(Dataset):
    def __init__(self, subjects_list, data_dir, chunk_duration=600, chunk_stride=300, 
                 downsample_freq=64, feature_columns=None, debug=False):
        """
        Args:
            subjects_list (list): List of subject IDs, e.g. ["SID1", "SID2", ...].
            data_dir (str): Directory containing files like "SID_whole_df.csv".
            chunk_duration (int): Chunk length in seconds.
            chunk_stride (int): Stride between chunks in seconds.
            downsample_freq (int): Target frequency after downsampling (from 64 Hz).
            debug (bool): Whether to print debugging information.
            feature_columns (list or None): List of columns to keep (e.g. ['TIMESTAMP', 'BVP', 'ACC', 'TEMP']).
                                          If None, a default list is used.
        """
        # Default features; note "ACC" is computed from the accelerometer axes.
        if feature_columns is None:
            self.feature_columns = ['ACC','TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI']
        else:
            self.feature_columns = feature_columns

        self.chunks = []
        self.downsample = int(64 // downsample_freq)
        self.chunk_length = int(chunk_duration * downsample_freq)
        self.stride = int(chunk_stride * downsample_freq)

        for SID in subjects_list:
            file_path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if os.path.exists(file_path):
                # Load the data. The converters dict can be kept from before if you use it for other columns.
                df = pd.read_csv(file_path, dtype={'Sleep_Stage': 'category'},
                                 converters=converters, low_memory=True)
                if debug:
                    print(f"Loaded data for subject {SID}")

                # Downsample: take every self.downsample-th row.
                if self.downsample != 1:
                    df = df.iloc[::self.downsample].reset_index(drop=True)
                    if debug:
                        print(f"After downsampling (factor {self.downsample}), rows: {len(df)}")

                # Remove rows in the "Preparation" phase labeled as 'P'.
                df = df[df['Sleep_Stage'] != 'P']

                # Convert to numeric for any columns we plan to use (except the computed ones)
                for col in df.columns:
                    if col in self.feature_columns and col != 'ACC':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # If 'ACC' is requested, compute it from the three accelerometer axes.
                if 'ACC' in self.feature_columns:
                    df['ACC'] = np.sqrt(df['ACC_X']**2 + df['ACC_Y']**2 + df['ACC_Z']**2)

                # Filter the dataframe to the columns of interest.
                df_X = df[self.feature_columns].copy()

                # Process sleep stage labels: trim whitespace and map to integer.
                df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
                df_Y = ( df['Sleep_Stage']
                    .map(SLEEP_STAGE_MAPPING)
                    .fillna(-1)          # everything unknown → “ignore”
                    .astype(int) )

                # Convert features and labels to numpy arrays.
                data_arr = df_X.values.astype(np.float32)
                labels_arr = df_Y.to_numpy()
                T = data_arr.shape[0]

                # If the record is shorter than one chunk, pad it.
                if T < self.chunk_length:
                    pad_size = self.chunk_length - T
                    padding_data = np.full((pad_size, data_arr.shape[1]), np.nan, dtype=np.float32)
                    data_arr = np.concatenate([data_arr, padding_data], axis=0)
                    padding_labels = np.full((pad_size,), -1)
                    labels_arr = np.concatenate([labels_arr, padding_labels], axis=0)
                    T = self.chunk_length

                # Create overlapping chunks using a sliding window.
                for start in range(0, T - self.chunk_length + 1, self.stride):
                    end = start + self.chunk_length
                    chunk_data = data_arr[start:end, :]
                    chunk_labels = labels_arr[start:end]
                    self.chunks.append({
                        'data': chunk_data,
                        'labels': chunk_labels,
                        'SID': SID
                    })
                if debug:
                    num_chunks = (T - self.chunk_length) // self.stride + 1
                    print(f"Subject {SID}: {T} samples processed, generated {num_chunks} chunks")
            else:
                print(f"File {file_path} does not exist. Skipping subject {SID}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        data = torch.tensor(chunk['data'], dtype=torch.float32)
        labels = torch.tensor(chunk['labels'], dtype=torch.long)
        # Forward fill to replace any NaN values with the previous valid value.
        data = forward_fill(data)
        return data, labels


# ### Construct train, val, and test datasets and dataloaders

# In[ ]:


'''
train_dataset_windowed = SleepDataset(subjects_list=subjects_train,
                                 data_dir=datadir_64Hz,
                                 window_size_ms= 20000, # 20 seconds of data
                                 stride_ms=5000,        # 5 seconds overlap
                                 downsample_freq=8, # downsample to 16Hz
                                 debug=False)
val_dataset_windowed = SleepDataset(subjects_list=subjects_val,
                                 data_dir=datadir_64Hz,
                                 window_size_ms= 20000, # 20 seconds of data
                                 stride_ms=5000,        # 5 seconds overlap
                                 downsample_freq=8, # downsample to 16Hz
                                 debug=False)
test_dataset_windowed = SleepDataset(subjects_list=subjects_test,
                                 data_dir=datadir_64Hz,
                                 window_size_ms= 20000, # 20 seconds of data
                                 stride_ms=5000,        # 5 second stride
                                 downsample_freq=8, # downsample to 16Hz
                                 debug=False)
'''
target_freq = 0.2
train_dataset = SleepDataset(subjects_list=subjects_train,
                                 data_dir=datadir_64Hz,
                                 max_length=max_length,
                                 downsample_freq=target_freq, # downsample to 8Hz
                                 debug=False)
print(f"Total samples in train dataset: {len(train_dataset)}")
val_dataset = SleepDataset(subjects_list=subjects_val,
                                 data_dir=datadir_64Hz,
                                 max_length=max_length,
                                 downsample_freq=target_freq, # downsample to 8Hz
                                 debug=False)
print(f"Total samples in val dataset: {len(val_dataset)}")                                 
test_dataset = SleepDataset(subjects_list=subjects_test,
                                 data_dir=datadir_64Hz,
                                 max_length=max_length,
                                 downsample_freq=target_freq, # downsample to 8Hz
                                 debug=False)
print(f"Total samples in test dataset: {len(test_dataset)}")


# In[ ]:


target_freq = 16
train_chunk_dataset = SleepChunkDataset(subjects_list=subjects_train,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=6000,  # 100 minutes
                                 chunk_stride=300,    # 5 minutes
                                 downsample_freq=target_freq,  
                                 feature_columns=['TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI', 'ACC'],
                                 debug=False)
print(f"Total samples in train chunk dataset: {len(train_chunk_dataset)}")
val_chunk_dataset = SleepChunkDataset(subjects_list=subjects_val,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=6000,  # 100 minutes
                                 chunk_stride=300,    # 5 minutes
                                 downsample_freq=target_freq, 
                                 feature_columns=['TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI', 'ACC'],
                                 debug=False)
print(f"Total samples in val chunk dataset: {len(val_chunk_dataset)}")
test_chunk_dataset = SleepChunkDataset(subjects_list=subjects_test,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=6000,  # 100 minutes
                                 chunk_stride=300,    # 5 minutes
                                 downsample_freq=target_freq, 
                                 feature_columns=['TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI', 'ACC'],
                                 debug=False)
print(f"Total samples in test chunk dataset: {len(test_chunk_dataset)}")


# In[ ]:


# get class weights for weighted loss
all_labels = []
for batch in DataLoader(train_chunk_dataset, batch_size=1):
    labels = batch[1].numpy()
    all_labels.extend(labels.flatten())
all_labels = np.array(all_labels)
valid_labels = all_labels[all_labels != -1]
classes = np.unique(valid_labels)
class_counts = Counter(valid_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=valid_labels
)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")
# Class weights: [0.82251347 5.00621272 0.49379841 1.78780618]


# In[ ]:


# CNN to downsample acceleration vector by a factor of 320


# ## CNN downsampling approach

# ### Model Definition

# In[ ]:


# sequence length x input channels -> CNN -> shortened sequence length x num hidden channels -> LSTM -> shortened sequence length x num sleep stages

class FeatureExtractorCNN(nn.Module):
    def __init__(self, in_channels=4, cnn_output_channels=128):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.convx = nn.Conv1d(64, cnn_output_channels, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm1d(cnn_output_channels)
        self.relu = nn.ReLU()
        # Global average pooling to collapse 
    
    def forward(self, x):
        assert not torch.isnan(x).any(), "NaN detected in CNN input"
        # Expect x of shape (batch, epoch_samples, channels)
        x = x.permute(0, 2, 1)  # Rearrange to (batch, channels, epoch_samples)
        #print(f"Input shape after permutation: {x.shape}")
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        #print(f"Shape after conv1 and pool1: {x.shape}")
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        #print(f"Shape after conv2 and pool2: {x.shape}")
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        #print(f"Shape after conv3 and pool3: {x.shape}")
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        #print(f"Shape after conv4 and pool4: {x.shape}")
        x = self.relu(self.conv5(x))
        x = self.pool5(x)
        #print(f"Shape after conv5 and pool5: {x.shape}")
        #x = self.relu(self.conv6(x))
        #x = self.pool6(x)
        #print(f"Shape after conv6 and pool6: {x.shape}")
        #x = self.relu(self.conv7(x))
        #x = self.pool7(x)
        #print(f"Shape after conv8 and pool8: {x.shape}")
        x = self.relu(self.convx(x))
        x = self.bn(x)
        return x

class SleepStageLSTM(nn.Module):
    def __init__(self, cnn_output_channels=128, hidden_size=64, num_layers=2, num_sleep_stages=5):
        super(SleepStageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=cnn_output_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False)
        self.fc = nn.Linear(hidden_size, num_sleep_stages)
    
    def forward(self, x):
        assert not torch.isnan(x).any(), "NaN detected in LSTM input"
        # x is of shape (batch, cnn_output_channels, samples)
        # LSTM expects input shape of (samples, batch, cnn_output_channels)
        x = x.permute(2, 0, 1) # (batch, cnn_output_channels, samples) -> (samples, batch, cnn_output_channels)
        #print(f"Input shape after permutation: {x.shape}")
        lstm_out, _ = self.lstm(x)
        #print(f"Shape after LSTM: {lstm_out.shape}")
        # Option 1: produce a prediction for every epoch (each time step)
        out = self.fc(lstm_out)   # shape: (batch, num_epochs, num_sleep_stages)
        #print(f"Shape after fully connected layer: {out.shape}")
        
        # Option 2: if you want a prediction only for the current epoch,
        # you may take the output of the last time step:
        #predictions = self.fc(lstm_out[:, -1, :])  # shape: (batch, num_sleep_stages)
        return out

class OnlineSleepStagingModel(pl.LightningModule):
    def __init__(self, in_channels, cnn_output_channels, lstm_hidden_size, num_layers=2, num_sleep_stages=5, learning_rate=0.001, class_weights=None):
        super(OnlineSleepStagingModel, self).__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractorCNN(in_channels=in_channels, cnn_output_channels=cnn_output_channels)
        self.lstm_model = SleepStageLSTM(cnn_output_channels=cnn_output_channels, hidden_size=lstm_hidden_size, num_layers=num_layers, num_sleep_stages=num_sleep_stages)
        self.learning_rate = learning_rate


        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore the "Missing" label (-1)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32), ignore_index=-1)
        self.num_sleep_stages = num_sleep_stages
        self.cnn_output_channels = cnn_output_channels

        self.val_class_counts = Counter()
        self.pred_class_counts = Counter()
        self.kappa = MulticlassCohenKappa(num_classes=self.num_sleep_stages)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.lstm_model(x) # (samples, batch_size, num_sleep_stages)
        assert x.shape[2] == self.num_sleep_stages, "LSTM output shape != num_sleep_stages"
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch  # y shape: [batch_size, T]
        y_hat = self(x)  # y_hat shape: [output_length, batch_size, num_sleep_stages]
        # Check for NaNs in the network output
        assert not torch.isnan(y).any(), "NaN detected in labels"
        assert not torch.isnan(y_hat).any(), "NaN detected in network output"
    
        # Permute to batch first
        y_hat = y_hat.permute(1, 0, 2)
        output_length = y_hat.shape[1]
        y_expanded = y.unsqueeze(1)
        # Downsample y to match y_hat
        y_resampled = torch.nn.functional.interpolate(
            y_expanded.float(),
            size = (output_length,),
            mode = 'nearest'
        )
        y_resampled = y_resampled.squeeze(1).long()

        # Flatten y_hat and y_resampled for loss calculation
        batch_size, output_length, num_sleep_stages = y_hat.shape
        y_hat_flat = y_hat.reshape(batch_size * output_length, num_sleep_stages)
        y_resampled_flat = y_resampled.reshape(batch_size * output_length)

        unique_labels = torch.unique(y_resampled_flat)
        assert ((unique_labels >= -1) & (unique_labels < self.num_sleep_stages)).all(), \
       "Found a label outside valid range!"


        # Calculate loss
        loss = self.criterion(y_hat_flat, y_resampled_flat)
        # Check loss for finiteness
        assert torch.isfinite(loss), "Loss is not finite"
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  # y shape: [batch_size, T]
        y_hat = self(x)  # y_hat shape: [output_length, batch_size, num_sleep_stages]
        # Permute to batch first
        y_hat = y_hat.permute(1, 0, 2)
        output_length = y_hat.shape[1]
        y_expanded = y.unsqueeze(1)
        # Downsample y to match y_hat
        y_resampled = torch.nn.functional.interpolate(
            y_expanded.float(),
            size = (output_length,),
            mode = 'nearest'
        )
        y_resampled = y_resampled.squeeze(1).long()

        # Flatten y_hat and y_resampled for loss calculation
        batch_size, output_length, num_sleep_stages = y_hat.shape
        y_hat_flat = y_hat.reshape(batch_size * output_length, num_sleep_stages)
        y_resampled_flat = y_resampled.reshape(batch_size * output_length)
        predictions = torch.argmax(y_hat_flat, dim=1)
        # Calculate Cohen's Kappa
        assert predictions.shape[0] == y_resampled_flat.shape[0], f"Predictions and labels have different shapes (dim 0) {predictions.shape[0]} vs {y_resampled_flat.shape[0]}"
        cohen_kappa_score = self.kappa(predictions, y_resampled_flat)
        self.log("val_cohen_kappa", cohen_kappa_score, prog_bar=True)
        # Calculate loss
        loss = self.criterion(y_hat_flat, y_resampled_flat)
        self.log("val_loss", loss, prog_bar=True)


        # Update class counts
        # y_resampled_flat: [batch_size * output_length]
        # predictions: [batch_size * output_length]

        mask = y_resampled_flat != 0
        y_valid = y_resampled_flat[mask]
        preds_valid = predictions[mask]

        if y_valid.numel() > 0:
            self.kappa.update(preds_valid, y_valid)
            self.val_class_counts.update(y_valid.cpu().tolist())
            self.pred_class_counts.update(preds_valid.cpu().tolist())

        return loss
  

    def on_validation_epoch_end(self):
        # Log Cohen's Kappa
        if self.kappa.confmat.sum() > 0:
            cohen_kappa_score = self.kappa.compute()
            self.log("val_cohen_kappa", cohen_kappa_score, prog_bar=True)
            self.kappa.reset()
        else:
            self.log("val_cohen_kappa", 0.0, prog_bar=True)

        # W&B class distribution bar plots
        class_labels = list(range(self.num_sleep_stages))
        val_counts = [self.val_class_counts.get(c, 0) for c in class_labels]
        pred_counts = [self.pred_class_counts.get(c, 0) for c in class_labels]

        for c in range(self.num_sleep_stages):
            self.log(f"class_count_true/{c}", self.val_class_counts.get(c, 0), on_epoch=True, prog_bar=False)
            self.log(f"class_count_pred/{c}", self.pred_class_counts.get(c, 0), on_epoch=True, prog_bar=False)

        self.val_class_counts.clear()
        self.pred_class_counts.clear()


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

class CNNClassifier(pl.LightningModule):
    def __init__(self, in_channels, cnn_output_channels, lstm_hidden_size, num_layers=2, num_sleep_stages=5, learning_rate=0.001):
        super(CNNClassifier, self).__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractorCNN(in_channels=in_channels, cnn_output_channels=cnn_output_channels)
        self.classifier = nn.Linear(cnn_output_channels, num_sleep_stages)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore the "Missing" label (-1)
        self.num_sleep_stages = num_sleep_stages
        self.cnn_output_channels = cnn_output_channels

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        assert x.shape[2] == self.num_sleep_stages, "CNN output shape != num_sleep_stages"
        return x



# ### Shape Demo

# In[ ]:


# demo of model elements
temp_input = train_chunk_dataset[0][0]
print(f"Input shape: {temp_input.shape} (epoch_samples, channels)")
CNN_model = FeatureExtractorCNN(in_channels=6, cnn_output_channels=128)
CNN_model.eval()
cnn_output = CNN_model(temp_input.unsqueeze(0))
print(f"Output shape: {cnn_output.shape} (batch_size, cnn_output_channels, epoch_samples)")
LSTM_model = SleepStageLSTM(cnn_output_channels=128, hidden_size=64, num_layers=2, num_sleep_stages=5)
LSTM_output = LSTM_model(cnn_output)

# demo of combined model
combined_model = OnlineSleepStagingModel(in_channels=6, cnn_output_channels=128, lstm_hidden_size=64, num_layers=2, num_sleep_stages=5)
combined_model.eval()
combined_output = combined_model(temp_input.unsqueeze(0))
print(f"Combined model output shape: {combined_output.shape} (samples, batch_size, num_sleep_stages)")
print(combined_output.shape[0])


# In[ ]:


print(temp_input.shape[0] / combined_output.shape[0])

wandb.finish()


# ### Hyperparameter Optimization

# In[ ]:


import optuna

def objective(trial):
    # Sample hyperparameters
    cnn_output_channels = trial.suggest_categorical("cnn_output_channels", [8, 16, 32, 64])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128])
    num_layers = 2
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    wandb_logger = WandbLogger(name=f"CNN{cnn_output_channels}_hs{lstm_hidden_size}_lr{learning_rate}", project="optuna_sleep_stage_classification")
    # DataLoaders (resample based on batch size)
    train_loader = DataLoader(train_chunk_dataset, batch_size=16, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_chunk_dataset, batch_size=16, num_workers=8, shuffle=False)

    # Model
    model = OnlineSleepStagingModel(
        in_channels=6,
        cnn_output_channels=cnn_output_channels,
        lstm_hidden_size=lstm_hidden_size,
        num_layers=num_layers,
        num_sleep_stages=5,
        learning_rate=learning_rate,
        class_weights=class_weights
    )

    # Trainer with pruning callback
    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator="gpu",
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, mode="min")
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    clear_output()
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

best_cnn_output_channels = best_trial.params["cnn_output_channels"]
best_lstm_hidden_size = best_trial.params["lstm_hidden_size"]
best_learning_rate = best_trial.params["learning_rate"]


# ### Model Training

# In[ ]:


# Train the combined model
wandb_logger = WandbLogger(project="sleep_stage_classification")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)
trainer = pl.Trainer(
    max_epochs=50,
    #fast_dev_run=True,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stop_callback]
)
model = OnlineSleepStagingModel(
    in_channels=6,
    cnn_output_channels=best_cnn_output_channels, 
    lstm_hidden_size=best_lstm_hidden_size, 
    num_layers=2, 
    num_sleep_stages=5, 
    learning_rate=best_learning_rate, 
    class_weights=weight_tensor) 

train_loader = DataLoader(train_chunk_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_chunk_dataset, batch_size=8, shuffle=False)
trainer.fit(model, train_loader, val_loader)
wandb.finish()
# Load the best model
best_model_path = checkpoint_callback.best_model_path
best_model = OnlineSleepStagingModel.load_from_checkpoint(best_model_path)


# ## CNN To Sleep Transformer approach
# 

# ### Dataset creation

# In[ ]:


downsample_freq = 64
train_chunk_dataset_noacc = SleepChunkDataset(subjects_list=subjects_train,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=1800,  # 30 minutes
                                 chunk_stride=300,    # 5 minutes
                                 downsample_freq=downsample_freq,   # freq to downsample to
                                 feature_columns=['TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI'],
                                 debug=False)
print(f"Total samples in train chunk dataset: {len(train_chunk_dataset)}")
val_chunk_dataset_noacc = SleepChunkDataset(subjects_list=subjects_val,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=1800,
                                 chunk_stride=300,    # 5 minutes
                                 downsample_freq=downsample_freq,   # freq to downsample to
                                 feature_columns=['TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI'],
                                 debug=False)
print(f"Total samples in val chunk dataset: {len(val_chunk_dataset)}")
test_chunk_dataset_noacc = SleepChunkDataset(subjects_list=subjects_test,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=1800,
                                 chunk_stride=300,    # 5 minutes
                                 downsample_freq=downsample_freq,   # freq to downsample to
                                 feature_columns=['TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI'],
                                 debug=False)
print(f"Total samples in test chunk dataset: {len(test_chunk_dataset)}")


# ### Model Definition

# In[ ]:


import math
import torch
from torch import nn
import pytorch_lightning as pl
from x_transformers import Encoder, Decoder, ContinuousTransformerWrapper

# Positional Encoding module.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]

class Seq2SeqSleepStager(pl.LightningModule):
    def __init__(self, input_dim, num_cnn_features, d_model, num_heads, num_enc_layers,
                 num_dec_layers, d_ff, dropout, num_classes, max_length, lr=1e-4, weight_tensor=None):
        """
        Args:
            input_dim (int): Number of continuous features per timestep.
            d_model (int): Transformer model dimension.
            num_heads (int): Number of attention heads.
            num_enc_layers (int): Number of encoder layers.
            num_dec_layers (int): Number of decoder layers.
            d_ff (int): Feed-forward dimension (e.g., 256).
            dropout (float): Dropout rate.
            num_classes (int): Number of sleep stage labels.
            max_length (int): Maximum sequence length.
            lr (float): Learning rate.
            weight_tensor (Tensor or None): Optional tensor of shape [vocab_size] for weighted loss.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['weight_tensor'])
        self.learning_rate = lr
        
        self.cnn = FeatureExtractorCNN(in_channels=input_dim, cnn_output_channels=num_cnn_features)
        # Continuous encoder: embeds continuous input.
        self.encoder = ContinuousTransformerWrapper(
            attn_layers=Encoder(
                dim=d_model,
                depth=num_enc_layers,
                heads=num_heads,
                ff_mult=d_ff // d_model,
                attn_dropout=dropout
            ),
            dim_in=num_cnn_features,
            max_seq_len=max_length
        )
        
        # For the decoder, assume sleep stage labels are discrete tokens.
        # E.g., if you have 5 sleep stages plus a special SOS token.
        vocab_size = num_classes + 1
        self.vocab_size = vocab_size
        self.decoder_embed = nn.Embedding(vocab_size, d_model)
        
        # Standard decoder with cross-attention enabled.
        self.decoder = Decoder(
            dim=d_model,
            depth=num_dec_layers,
            heads=num_heads,
            ff_mult=d_ff // d_model,
            attn_dropout=dropout,
            cross_attend=True
        )
        
        # Positional encoding for the decoder.
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_length)
        
        # Linear projection to convert decoder output to vocab logits.
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Use cross-entropy loss; pass weight_tensor if provided.
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=0)
    
    def forward(self, src, tgt):
        """
        Args:
            src (Tensor): Continuous input tensor of shape [B, src_len, input_dim].
            tgt (Tensor): Target token sequence [B, tgt_len]. For training, this should be teacher-forced (shifted right).
        Returns:
            logits (Tensor): Output logits of shape [B, tgt_len, vocab_size].
        """
        unique_vals = torch.unique(tgt)
        #print(f"Unique target token indices: {unique_vals}")
        assert torch.all(tgt >= 0) and torch.all(tgt < self.vocab_size), "Target indices out of range!"

        # Encode the continuous source data.
        memory = self.encoder(src)  # [B, src_len, d_model]
        
        # Embed target tokens and add positional encodings.
        dec_inp = self.decoder_embed(tgt)  # [B, tgt_len, d_model]
        dec_inp = self.positional_encoding(dec_inp)
        
        # Decode using memory as context.
        dec_out = self.decoder(dec_inp, context=memory)  # [B, tgt_len, d_model]
        logits = self.out_proj(dec_out)  # [B, tgt_len, vocab_size]
        return logits
    
    def training_step(self, batch, batch_idx):
        # Expect batch to be a tuple: (src, tgt).
        src, tgt = batch
        # Teacher forcing: decoder input is tgt[:, :-1], target is tgt[:, 1:].
        logits = self(src, tgt[:, :-1])
        target = tgt[:, 1:]
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt[:, :-1])
        target = tgt[:, 1:]
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# In[ ]:


src,tgt = next(iter(DataLoader(train_chunk_dataset_noacc, batch_size=8)))
print(f"Source shape: {src.shape} (batch_size, epoch_samples, channels)")
print(f"Target shape: {tgt.shape} (batch_size, epoch_samples)")


# ### Model Training

# In[ ]:


print(f"Max length: {max_length}")
print(f"Max length after downsampling: {int(max_length // (64 // 0.2))}")
wandb.finish()


# In[ ]:


# Train the transformer
wandb_logger = WandbLogger(name="online_sleep_staging_model", project="sleep_stage_transformer")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)
trainer = pl.Trainer(
    max_epochs=50,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    precision=16,
    gradient_clip_val=1.0,
    callbacks=[checkpoint_callback, early_stop_callback]
)
model = Seq2SeqSleepStager(
    input_dim=5,
    num_cnn_features = 32
    d_model=128,
    num_heads=4, 
    num_enc_layers=2, 
    num_dec_layers=2, 
    d_ff=256, 
    dropout=0.1, 
    num_classes=5, 
    max_length= int(max_length // (64 // 64)),  # Adjusted for downsampling
    lr=1e-3,
    #weight_tensor=torch.tensor(class_weights, dtype=torch.float32)
)

train_loader = DataLoader(train_chunk_dataset_noacc, batch_size=16, shuffle=True)
val_loader = DataLoader(val_chunk_dataset_noacc, batch_size=16, shuffle=False)
trainer.fit(model, train_loader, val_loader)
wandb.finish()
# Load the best model
best_model_path = checkpoint_callback.best_model_path


# ## ACC aware CNN

# ### Mixed Frequency Dataset Class

# In[ ]:


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
    "N2": 2,   # non-REM stage 2
    "N3": 3,   # non-REM stage 3
    "R": 4,    # REM
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


class MixedFreqDataset(Dataset):
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
            self.feature_columns = ['ACC', 'TIMESTAMP', 'BVP', 'TEMP', 'HR', 'IBI']
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
            if c != 'ACC'
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

            # 2) compute & downsample ACC
            df['ACC'] = np.sqrt(
                df['ACC_X']**2 + df['ACC_Y']**2 + df['ACC_Z']**2
            )
            if self.downsample_acc != 1:
                df = df.iloc[::self.downsample_acc].reset_index(drop=True)
                if debug:
                    print(f"[DEBUG] {SID}: ACC ↓ to {int(64/self.downsample_acc)} Hz → {len(df)} rows")
            acc_arr = df['ACC'].values.astype(np.float32)

            # 3) downsample *all* channels for non‐acc view
            df = df.iloc[::self.downsample].reset_index(drop=True)
            if debug:
                print(f"[DEBUG] {SID}: non-ACC ↓ to {int(64/self.downsample)} Hz → {len(df)} rows")

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
            # 5) assemble feature matrix
            data_arr = df[self.feature_columns].values.astype(np.float32)

            # 6) pad short records
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

            # 7) slice into overlapping chunks
            for start in range(0, T - self.chunk_length + 1, self.stride):
                end       = start + self.chunk_length
                start_acc = int(start * self.ratio)
                end_acc   = start_acc + self.chunk_length_acc

                non_acc_chunk = data_arr[start:end, self.non_acc_idxs]
                acc_chunk     = acc_arr[start_acc:end_acc]
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
        non_acc = torch.tensor(c['non_acc'], dtype=torch.float32)
        acc     = torch.tensor(c['acc'],     dtype=torch.float32)
        labels  = torch.tensor(c['labels'],  dtype=torch.long)

        # forward‑fill each
        non_acc = forward_fill(non_acc)
        acc     = forward_fill(acc.unsqueeze(1)).squeeze(1)

        return non_acc, acc, labels


# ### Construct mixed frequency datasets

# In[ ]:


non_acc_freq = 0.2
acc_freq = 32
chunk_duration = 12000
chunk_stride = 6000
train_dataset_mixed = MixedFreqDataset(subjects_list=subjects_train,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=chunk_duration,
                                 chunk_stride=chunk_stride,
                                 downsample_freq=non_acc_freq, # downsample to 8Hz
                                 acc_freq=acc_freq,
                                 debug=False)
print(f"Total samples in train dataset: {len(train_dataset_mixed)}")
val_dataset_mixed = MixedFreqDataset(subjects_list=subjects_val,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=chunk_duration,
                                 chunk_stride=chunk_stride,
                                 downsample_freq=non_acc_freq, # downsample to 8Hz
                                 acc_freq=acc_freq,
                                 debug=False)
print(f"Total samples in val dataset: {len(val_dataset_mixed)}")
test_dataset_mixed = MixedFreqDataset(subjects_list=subjects_test,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=chunk_duration,
                                 chunk_stride=chunk_stride,
                                 downsample_freq=non_acc_freq, # downsample to 8Hz
                                 acc_freq=acc_freq,
                                 debug=False)
print(f"Total samples in test dataset: {len(test_dataset_mixed)}")

train_dataset_mixed_small = MixedFreqDataset(subjects_list=subjects_train_small,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=chunk_duration,
                                 chunk_stride=chunk_stride,
                                 downsample_freq=non_acc_freq, # downsample to 8Hz
                                 acc_freq=acc_freq,
                                 debug=False)
print(f"Total samples in train dataset small: {len(train_dataset_mixed_small)}")
val_dataset_mixed_small = MixedFreqDataset(subjects_list=subjects_val_small,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=chunk_duration,
                                 chunk_stride=chunk_stride,
                                 downsample_freq=non_acc_freq, # downsample to 8Hz
                                 acc_freq=acc_freq,
                                 debug=False)
print(f"Total samples in val dataset small: {len(val_dataset_mixed_small)}")
test_dataset_mixed_small = MixedFreqDataset(subjects_list=subjects_test_small,
                                 data_dir=datadir_64Hz,
                                 chunk_duration=chunk_duration,
                                 chunk_stride=chunk_stride,
                                 downsample_freq=non_acc_freq, # downsample to 8Hz
                                 acc_freq=acc_freq,
                                 debug=False)
print(f"Total samples in test dataset small: {len(test_dataset_mixed_small)}")


# In[ ]:


class ACCFeatureExtractorCNN(nn.Module):
    def __init__(self, output_channels=16):
        super(ACCFeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=512, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=256, stride=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=256, stride=2)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=output_channels, kernel_size=32, stride=2)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(output_channels)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        assert not torch.isnan(x).any(), "NaN detected in CNN input"
        # Expect x of shape (batch, epoch_samples, channels)
        x = x.permute(0, 2, 1)  # (batch, channels (1), epoch_samples)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        return x


class ACCAwareSleepStager(pl.LightningModule):
    def __init__(self, non_acc_dim: int,
                 cnn_output_channels: int = 16,
                 lstm_hidden_size:   int = 64,
                 lstm_layers:        int = 2,
                 num_sleep_stages:   int = 5,
                 lr:                  float = 1e-3,
                 weight_tensor:      torch.Tensor = None,
                 debug:            bool = False,):
        super().__init__()
        self.save_hyperparameters()

        self.acc_cnn = ACCFeatureExtractorCNN(cnn_output_channels)

        self.lstm_input_size = cnn_output_channels + non_acc_dim
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=False)
        self.classifier = nn.Linear(lstm_hidden_size, num_sleep_stages)

        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)
        self.lr        = lr
        self.debug      = debug
        self.kappa = MulticlassCohenKappa(num_classes=num_sleep_stages)

    
    def forward(self, non_acc, acc):
        """
        non_acc: (batch, non_acc_length, non_acc_dim)
        acc:     (batch, acc_length)  or  (batch, acc_length, 1)
        returns:
          y_hat (output_length, batch, num_sleep_stages)
        """
        # ensure ACC has a channel dim
        if acc.dim() == 2:
            acc = acc.unsqueeze(-1)          # now (batch, T, 1)

        # 1) ACC → CNN → (batch, num_cnn_feats, cnn_output_length)
        acc_feats = self.acc_cnn(acc)
        if self.debug:
            print(f"[DEBUG] ACC CNN output shape: {acc_feats.shape}") # (batch, num_cnn_feats, cnn_output_length)
            print(f"[DEBUG] NON_ACC input shape: {non_acc.shape}") # (batch, non_acc_length, non_acc_dim)


        # 2) downsample whichever sequence is longer
        cnn_output_length = acc_feats.shape[2]
        non_acc_output_length = non_acc.shape[1]
        if cnn_output_length > non_acc_output_length:
            if self.debug:
                print(f"[DEBUG] Downsampling ACC features from {cnn_output_length} to {non_acc_output_length}")
            acc_feats = F.interpolate(
                acc_feats,
                size=non_acc_output_length)
        else:
            if self.debug:
                print(f"[DEBUG] Downsampling non-ACC features from {non_acc_output_length} to {cnn_output_length}")
            non_acc = F.interpolate(
                non_acc.permute(0,2,1),  # (batch, non_acc_dim, non_acc_length) (for interpolate function syntax)
                size=cnn_output_length)
            non_acc = non_acc.permute(0,2,1)  # (batch, non_acc_length, non_acc_dim) (switching back)

        if self.debug:
            print(f"[DEBUG] ACC features shape: {acc_feats.shape}")
            print(f"[DEBUG] non-ACC features shape: {non_acc.shape}")

        # 3) build LSTM input: (T', batch, feature_dim)
        a = acc_feats.permute(2, 0, 1)        # (lstm_seq_len, batch, cnn_output_features)
        b = non_acc.permute(1, 0, 2)       # (lstm_seq_len, batch, non_acc_dim)
        lstm_in = torch.cat([a, b], dim=2)    # (lstm_seq_len, batch, C_cnn + D_nonacc)
        # without batch_first = true, LSTM input shape is (seq_len, batch_size, features)

        if self.debug:
            print(f"[DEBUG] LSTM input shape: {lstm_in.shape}")

        # 4) LSTM + classifier
        lstm_out, _ = self.lstm(lstm_in)      # (lstm_seq_len, batch, lstm_hidden_size)
        if self.debug:
            print(f"[DEBUG] LSTM output shape: {lstm_out.shape}")
        y_hat = self.classifier(lstm_out)     # (lstm_seq_len, batch, num_sleep_stages)
        if self.debug:
            print(f"[DEBUG] Classifier output shape: {y_hat.shape}")
        return y_hat

    def training_step(self, batch, batch_idx):
        non_acc, acc, labels = batch
        '''
        non_acc: (batch, non_acc_length, non_acc_dim)
        acc:     (batch, acc_length)  or  (batch, acc_length, 1)
        labels:  (batch, non_acc_length)
        '''
        y_hat = self(non_acc, acc)            # (lstm_seq_len, batch, num_sleep_stages)
        y_hat = y_hat.permute(1, 0, 2)        # (batch, lstm_seq_len, num_sleep_stages)


        # flatten
        batch_size, output_length, num_sleep_stages = y_hat.shape
        y_hat_flat = y_hat.reshape(batch_size * output_length, num_sleep_stages)
        labels_flat  = labels.reshape(batch_size * output_length)
        
        # calculate loss
        loss = self.criterion(y_hat_flat, labels_flat)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        non_acc, acc, labels = batch
        '''
        non_acc: (batch, non_acc_length, non_acc_dim)
        acc:     (batch, acc_length)  or  (batch, acc_length, 1)
        labels:  (batch, non_acc_length)
        '''
        y_hat = self(non_acc, acc)            # (lstm_seq_len, batch, num_sleep_stages)
        y_hat = y_hat.permute(1, 0, 2)        # (batch, lstm_seq_len, num_sleep_stages)

        # flatten
        batch_size, output_length, num_sleep_stages = y_hat.shape
        y_hat_flat = y_hat.reshape(batch_size * output_length, num_sleep_stages)
        labels_flat  = labels.reshape(batch_size * output_length)
        
        # calculate loss
        loss = self.criterion(y_hat_flat, labels_flat)

        # calculate accuracy
        predictions = torch.argmax(y_hat_flat, dim=1)
        mask = labels_flat != -1
        masked_preds = predictions[mask]
        masked_labels = labels_flat[mask]
        if masked_labels.numel() > 0:
            acc = (masked_preds == masked_labels).float().mean().item()
        else:
            acc = 0.0

        # calculate cohen's kappa
        mask = labels_flat != -1
        y_valid = labels_flat[mask]
        preds_valid = predictions[mask]
        if y_valid.numel() > 0:
            ckappa = self.kappa(preds_valid, y_valid)
        else:
            ckappa = torch.tensor(0.0, device=self.device)

        # log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)
        self.log("val_cohen_kappa", ckappa, prog_bar=True)

        self.kappa.reset()

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# In[ ]:


# Demo
temp_non_acc, temp_acc, temp_labels = train_dataset_mixed[0]
print(f"Non-acc shape: {temp_non_acc.shape}")
print(f"ACC shape: {temp_acc.shape}")
print(f"Labels shape: {temp_labels.shape}")

CNN_model = ACCFeatureExtractorCNN(output_channels=16)
CNN_model.eval()
cnn_output = CNN_model(temp_acc.unsqueeze(0).unsqueeze(2)) # add batch and channel dimensions
print(f"Output shape: {cnn_output.shape} (batch_size, cnn_output_channels, epoch_samples)")
print(f"CNN downsampling factor: {temp_acc.shape[0] / cnn_output.shape[2]}")

model = ACCAwareSleepStager(
    non_acc_dim=temp_non_acc.shape[1],
    cnn_output_channels=16,
    lstm_hidden_size=64,
    lstm_layers=2,
    num_sleep_stages=5,
    lr=1e-3,
    debug=True
)
out = model(temp_non_acc.unsqueeze(0), temp_acc.unsqueeze(0).unsqueeze(2))


# ### Get Class Weights

# In[ ]:


# get class weights for weighted loss
all_labels = []
for batch in DataLoader(train_dataset_mixed, batch_size=1):
    labels = batch[2].numpy()
    all_labels.extend(labels.flatten())
all_labels = np.array(all_labels)
valid_labels = all_labels[all_labels != -1]
classes = np.unique(valid_labels)
class_counts = Counter(valid_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=valid_labels
)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")


# ### Hyperparameter Optimization

# In[ ]:


import optuna

def objective(trial):
    # Sample hyperparameters
    cnn_output_channels = trial.suggest_categorical("cnn_output_channels", [8, 16, 32, 64, 128])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256])
    num_layers = 2
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    wandb_logger = WandbLogger(name=f"nf_{cnn_output_channels}hs_{lstm_hidden_size}lr_{learning_rate}", project="acc_aware_model")
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
    )
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
    )
    trainer = pl.Trainer(
    max_epochs=20,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    log_every_n_steps=2,
    )
    model = ACCAwareSleepStager(
    non_acc_dim=5,
    cnn_output_channels=cnn_output_channels,
    lstm_hidden_size=lstm_hidden_size,
    lstm_layers=2,
    num_sleep_stages=5,
    lr=learning_rate,
    weight_tensor=torch.tensor(class_weights, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset_mixed_small, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset_mixed_small, batch_size=8, shuffle=False)
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

best_cnn_output_channels = best_trial.params["cnn_output_channels"]
best_lstm_hidden_size = best_trial.params["lstm_hidden_size"]
best_learning_rate = best_trial.params["learning_rate"]


# ### Train Model

# In[ ]:


# Train the acc aware model
wandb_logger = WandbLogger(project="acc_aware_model")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)
trainer = pl.Trainer(
    max_epochs=100,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stop_callback]
)
model = ACCAwareSleepStager( # cnn_output 32, lstm_hidden 128, lr 1e-4 did ok
    non_acc_dim=5,
    cnn_output_channels=best_cnn_output_channels,
    lstm_hidden_size=best_lstm_hidden_size,
    lstm_layers=2,
    num_sleep_stages=5,
    lr=best_learning_rate,
    weight_tensor=torch.tensor(class_weights, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset_mixed, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset_mixed, batch_size=8, shuffle=False)
trainer.fit(model, train_loader, val_loader)
wandb.finish()
# Load the best model
best_model_path = checkpoint_callback.best_model_path

