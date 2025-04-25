#!/usr/bin/env python
# coding: utf-8

# # Library Imports

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

torch.set_float32_matmul_precision('medium')


# # Data Loading

# ## Dataset Class

# In[ ]:


# Helper to safely convert strings to floats
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# Mapping sleep-stage labels to integers


class DualFreqDataset(Dataset):
    def __init__(self,
                 subjects_list,
                 data_dir,
                 chunk_duration: float = 600,
                 chunk_stride: float = 300,
                 high_freq: int = 32,
                 low_freq: int = 8,
                 hf_features: list = None,
                 lf_features: list = None,
                 debug: bool = False):
        self.hf_downsample = int(64 // high_freq) # downsample factor for high frequency data
        self.lf_downsample = int(64 // low_freq) # downsample factor for low frequency data

        SLEEP_STAGE_MAPPING = {
            "W": 0,    # Wake
            "N1": 1,   # non-REM stage 1
            "N2": 2,   # non-REM stage 2
            "N3": 3,   # non-REM stage 3
            "R": 4,    # REM
            "Missing": -1  # Missing label → ignore
        }
        numeric_columns = ['TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR', 'IBI']
        converters = {col: safe_float for col in numeric_columns}

        self.chunks = []
        for SID in subjects_list:
            path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist.")
            # Load data for subject
            df = pd.read_csv(path,
                             dtype={'Sleep_Stage': 'category'},
                             converters=converters,
                             low_memory=True)
            
            # drop preparation phase, map labels
            df = df[df['Sleep_Stage'] != 'P']
            df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
            labels_arr = (
                df['Sleep_Stage']
                  .map(SLEEP_STAGE_MAPPING)
                  .fillna(-1)
                  .astype(int)
                  .to_numpy()
            )
            # separate high and low frequency data
            df_high = df[hf_features].copy()
            df_low = df[lf_features].copy()
            # downsample data and labels
            df_high = df_high.iloc[::self.hf_downsample, :].reset_index(drop=True)
            df_low = df_low.iloc[::self.lf_downsample, :].reset_index(drop=True)
            labels_arr = labels_arr[::self.lf_downsample]
            # forward fill features
            df_high.ffill(inplace=True)
            df_high.bfill(inplace=True)
            df_low.ffill(inplace=True)
            df_low.bfill(inplace=True)
            # normalize data
            df_high = (df_high - df_high.mean()) / (df_high.std().replace(0, 1e-6))
            df_low = (df_low - df_low.mean()) / (df_low.std().replace(0, 1e-6))
            # create chunks
            total_time = int(len(df_high) / high_freq)
            n_chunks = int((total_time - chunk_duration) // chunk_stride) + 1
            for i in range(n_chunks):
                start_time = i * chunk_stride
                end_time = start_time + chunk_duration
                
                start_low = int(start_time * low_freq)
                end_low = int(end_time * low_freq)
                start_high = int(start_time * high_freq)
                end_high = int(end_time * high_freq)

                lf = df_low .iloc[start_low: end_low ].values.astype(np.float32)
                hf = df_high.iloc[start_high:end_high].values.astype(np.float32)
                labels = labels_arr[start_low: end_low]

                self.chunks.append({
                    'high': hf,
                    'low': lf,
                    'labels': labels,
                })
        if debug:
            print(f"Loaded {len(self.chunks)} chunks from {len(subjects_list)} subjects.")
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        hf = torch.tensor(chunk['high'], dtype=torch.float32)
        lf = torch.tensor(chunk['low'], dtype=torch.float32)
        labels = torch.tensor(chunk['labels'], dtype=torch.long)
        return hf, lf, labels


# ## Separate subjects into train/val/test

# In[ ]:


datadir_64Hz = '/gpfs/data/oermannlab/users/slj9342/dl4med_25/data/physionet.org/files/dreamt/2.0.0/data_64Hz/' # working with 64Hz data
max_length = 2493810 # found experimentally, takes a while to compute

participant_info_df = pd.read_csv('/gpfs/data/oermannlab/users/slj9342/dl4med_25/data/physionet.org/files/dreamt/2.0.0/participant_info.csv')
subjects_all = participant_info_df['SID']

subjects_all_shuffled = participant_info_df['SID'].sample(frac=1, random_state=42).reset_index(drop=True)
subjects_train = subjects_all_shuffled[:int(len(subjects_all_shuffled)*0.8)]
subjects_val = subjects_all_shuffled[int(len(subjects_all_shuffled)*0.8):int(len(subjects_all_shuffled)*0.9)]
subjects_test = subjects_all_shuffled[int(len(subjects_all_shuffled)*0.9):]
print(f"number of subjects in train: {len(subjects_train)}")
print(f"number of subjects in val: {len(subjects_val)}")
print(f"number of subjects in test: {len(subjects_test)}")

fraction = 0.3
subjects_train_small = subjects_train[:int(len(subjects_train)*fraction)]
subjects_val_small = subjects_val[:int(len(subjects_val)*fraction)]
subjects_test_small = subjects_test[:int(len(subjects_test)*fraction)]
print(f"number of subjects in small train: {len(subjects_train_small)}")
print(f"number of subjects in small val: {len(subjects_val_small)}")
print(f"number of subjects in small test: {len(subjects_test_small)}")


# ## Construct train, val, and test datasets

# In[35]:


hf_features = ['TIMESTAMP','BVP','ACC_X','ACC_Y','ACC_Z']
lf_features = ['TIMESTAMP','TEMP','EDA','HR','IBI']
hf_freq = 32
lf_freq = 0.2
chunk_duration = 6000 # 100 minutes
chunk_stride = 1500 # 25 minutes
train_dataset = DualFreqDataset(subjects_list=subjects_train,
                                data_dir=datadir_64Hz,
                                chunk_duration=chunk_duration,
                                chunk_stride=chunk_stride,
                                high_freq=hf_freq,
                                low_freq=lf_freq,
                                hf_features=hf_features,
                                lf_features=lf_features)
print(f"number of chunks in train: {len(train_dataset)}")
val_dataset = DualFreqDataset(subjects_list=subjects_val,
                              data_dir=datadir_64Hz,
                              chunk_duration=chunk_duration,
                              chunk_stride=chunk_stride,
                              high_freq=hf_freq,
                              low_freq=lf_freq,
                              hf_features=hf_features,
                              lf_features=lf_features)
print(f"number of chunks in val: {len(val_dataset)}")
test_dataset = DualFreqDataset(subjects_list=subjects_test,
                               data_dir=datadir_64Hz,
                               chunk_duration=chunk_duration,
                               chunk_stride=chunk_stride,
                               high_freq=hf_freq,
                               low_freq=lf_freq,
                               hf_features=hf_features,
                               lf_features=lf_features)
print(f"number of chunks in test: {len(test_dataset)}")
train_dataset_small = DualFreqDataset(subjects_list=subjects_train_small,
                                       data_dir=datadir_64Hz,
                                       chunk_duration=chunk_duration,
                                       chunk_stride=chunk_stride,
                                       high_freq=hf_freq,
                                       low_freq=lf_freq,
                                       hf_features=hf_features,
                                       lf_features=lf_features)
print(f"number of chunks in small train: {len(train_dataset_small)}")
val_dataset_small = DualFreqDataset(subjects_list=subjects_val_small,
                                     data_dir=datadir_64Hz,
                                     chunk_duration=chunk_duration,
                                     chunk_stride=chunk_stride,
                                     high_freq=hf_freq,
                                     low_freq=lf_freq,
                                     hf_features=hf_features,
                                     lf_features=lf_features)
print(f"number of chunks in small val: {len(val_dataset_small)}")
test_dataset_small = DualFreqDataset(subjects_list=subjects_test_small,
                                      data_dir=datadir_64Hz,
                                      chunk_duration=chunk_duration,
                                      chunk_stride=chunk_stride,
                                      high_freq=hf_freq,
                                      low_freq=lf_freq,
                                      hf_features=hf_features,
                                      lf_features=lf_features)
print(f"number of chunks in small test: {len(test_dataset_small)}")


# # Model Definition

# ## TCN

# In[36]:


class TemporalBlock(nn.Module):
    def __init__(self,
     input_channels, output_channels, kernel_size, dilation, stride=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation)
        self.bn1   = nn.BatchNorm1d(output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size,
                               stride=1, padding=padding,
                               dilation=dilation)
        self.bn2   = nn.BatchNorm1d(output_channels)
        self.relu  = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)
        # 1×1 conv to match channels/stride if needed
        self.downsample = (nn.Conv1d(input_channels, output_channels, 1, stride=stride)
                           if (stride!=1 or input_channels!=output_channels) else None)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class HFFeatureExtractorTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=5,
                 kernel_size=3,
                 base_channels=32,
                 final_down=64,     # match CNN’s total downsample factor
                 dropout=0.1):
        super().__init__()
        layers = []
        ch = in_channels
        # build dilated residual blocks (no downsampling here)
        for i in range(num_blocks):
            layers.append(
                TemporalBlock(ch, base_channels,
                              kernel_size=kernel_size,
                              dilation=2**i,
                              stride=1,
                              dropout=dropout)
            )
            ch = base_channels
        # final 1×1 conv with stride=final_down to downsample by 64
        layers.append(nn.Conv1d(ch, out_channels,
                                kernel_size=1,
                                stride=final_down,
                                padding=0))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        x = x.permute(0,2,1)  # → (batch, channels, seq_len)
        y = self.tcn(x)       # → (batch, out_channels, seq_len/64)
        return y             # leave it in (B, C, T’) form


# ## CNN

# In[37]:


class HFFeatureExtractorCNN(nn.Module):
    def __init__(self,
     input_channels=5,
     output_channels=16, 
     hidden_channels=32, 
     dropout=0.1):
        super(HFFeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=512, stride=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=256, stride=2)
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=256, stride=2)
        self.conv4 = nn.Conv1d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=32, stride=2)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(output_channels)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(dropout)
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


# ## Combined Model

# In[38]:


class DualFreqSleepStager(pl.LightningModule):
    def __init__(self,
                 hf_input_channels=5,
                 lf_input_channels=5,
                 cnn_output_channels=16,
                 cnn_hidden_channels=32,
                 lstm_hidden_size=64,
                 lstm_num_layers=2,
                 lstm_bidirectional=True,
                 dropout=0.1,
                 num_sleep_stages=5,
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 weight_tensor=None,
                 convnet='CNN',
                 debug=False):
        super().__init__()
        self.save_hyperparameters()
        if convnet == 'CNN':
            self.cnn = HFFeatureExtractorCNN(input_channels=hf_input_channels,
                                        output_channels=cnn_output_channels,
                                        hidden_channels=cnn_hidden_channels,
                                        dropout=dropout)
        elif convnet == 'TCN':
            self.cnn = HFFeatureExtractorTCN(in_channels=hf_input_channels,
                                            out_channels=cnn_output_channels,
                                            num_blocks=5,
                                            kernel_size=3,
                                            base_channels=cnn_hidden_channels,
                                            final_down=64,     # match CNN’s total downsample factor
                                            dropout=dropout)
        else:
            raise ValueError(f"Unknown convnet type: {convnet}")
        self.lstm = nn.LSTM(
                            input_size=cnn_output_channels + lf_input_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            dropout=dropout,
                            batch_first=False)
        
        if lstm_bidirectional:
            self.classifier = nn.Linear(lstm_hidden_size * 2, num_sleep_stages, )
        else:
            self.classifier = nn.Linear(lstm_hidden_size, num_sleep_stages)

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.kappa = MulticlassCohenKappa(num_classes=num_sleep_stages)
        self.debug = debug
        

        if weight_tensor is not None:
            assert weight_tensor.shape[0] == num_sleep_stages, \
                f"Weight tensor shape {weight_tensor.shape[0]} does not match number of sleep stages {num_sleep_stages}"
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)

    def forward(self, hf, lf):
        assert not torch.isnan(hf).any(), "NaN detected in CNN input"
        assert not torch.isnan(lf).any(), "NaN detected in LSTM input"
        if self.debug:
            print(f"HF input shape: {hf.shape}")
            print(f"LF input shape: {lf.shape}")
        # pass high frequency data through CNN    
        cnn_features = self.cnn(hf)
        if self.debug:
            print(f"cnn output shape: {cnn_features.shape}")

        # downsample longer sequence
        cnn_output_length = cnn_features.shape[2]
        lf_output_length = lf.shape[1]
        if cnn_output_length > lf_output_length:
            if self.debug:
                print(f"[DEBUG] cnn output length {cnn_output_length} > lf output length {lf_output_length}, downsampling")
            cnn_features = F.interpolate(
                cnn_features,
                size=lf_output_length,
            )
        elif cnn_output_length < lf_output_length:
            if self.debug:
                print(f"[DEBUG] cnn output length {cnn_output_length} < lf output length {lf_output_length}, downsampling")
            lf = F.interpolate(
                lf,
                size=cnn_output_length,
            )
        if self.debug:
            print(f"[DEBUG] hf features shape: {cnn_features.shape}")
            print(f"[DEBUG] lf features shape: {lf.shape}")
        
        # concatenate high and low frequency features
        a = cnn_features.permute(2,0,1) # (sequence_length, batch_size, cnn_output_channels)
        b = lf.permute(1,0,2) # (sequence_length, batch_size, lf_input_channels)
        x = torch.cat((a, b), dim=2)
        if self.debug:
            print(f"[DEBUG] lstm input shape: {x.shape}")
        
        # pass through LSTM + classifier
        x, _ = self.lstm(x)
        if self.debug:
            print(f"[DEBUG] lstm output shape: {x.shape}")
        x = self.classifier(x)
        if self.debug:
            print(f"[DEBUG] classifier output shape: {x.shape}")
        return x
    def training_step(self, batch, batch_idx):
        hf, lf, labels = batch
        if self.debug:
            print(f"[DEBUG] training step batch {batch_idx}")
            print(f"[DEBUG] hf shape: {hf.shape}")
            print(f"[DEBUG] lf shape: {lf.shape}")
            print(f"[DEBUG] labels shape: {labels.shape}")
        
        logits = self(hf, lf)
        logits = logits.permute(1, 0, 2) # should be (batch_size, seq_len, num_classes)
        if self.debug:
            print(f"[DEBUG] logits shape after permute: {logits.shape}")

        # flatten
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, num_classes)
        labels_flat = labels.reshape(batch_size * seq_len)

        # calculate loss
        loss = self.criterion(logits_flat, labels_flat)
        if self.debug:
            print(f"[DEBUG] loss: {loss.item()}")

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        hf, lf, labels = batch
        if self.debug:
            print(f"[DEBUG] validation step batch {batch_idx}")
            print(f"[DEBUG] hf shape: {hf.shape}")
            print(f"[DEBUG] lf shape: {lf.shape}")
            print(f"[DEBUG] labels shape: {labels.shape}")

        logits = self(hf, lf)
        logits = logits.permute(1, 0, 2)
        if self.debug:
            print(f"[DEBUG] logits shape after permute: {logits.shape}")
        # flatten
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, num_classes)
        labels_flat = labels.reshape(batch_size * seq_len)
        if self.debug:
            print(f"[DEBUG] logits_flat shape: {logits_flat.shape}")
            print(f"[DEBUG] labels_flat shape: {labels_flat.shape}")
        # calculate loss
        loss = self.criterion(logits_flat, labels_flat)
        if self.debug:
            print(f"[DEBUG] validation loss: {loss.item()}")
        # calculate accuracy
        preds = torch.argmax(logits_flat, dim=1)
        mask = labels_flat != -1
        masked_preds = preds[mask]
        masked_labels = labels_flat[mask]
        if masked_labels.numel() > 0:
            acc = (masked_preds == masked_labels).float().mean().item()
        else:
            acc = 0.0
        if self.debug:
            print(f"[DEBUG] validation accuracy: {acc}")
        # calculate kappa
        kappa = self.kappa.update(masked_preds, masked_labels)
        if self.debug:
            print(f"[DEBUG] validation kappa: {kappa}")

        # log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return {
            'val_loss': loss,
            'val_acc': acc
        }

    def on_validation_epoch_end(self):
        kappa = self.kappa.compute()
        self.log('val_cohen_kappa', torch.nan_to_num(kappa,0.0), prog_bar=True)
        self.kappa.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        '''
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
        '''
        return optimizer


# ## Model Demo (shape compatibility)

# In[39]:


temp_hf, temp_lf, temp_labels = train_dataset[0]
print(f"temp_hf shape: {temp_hf.shape}")
print(f"temp_lf shape: {temp_lf.shape}")
print(f"temp_labels shape: {temp_labels.shape}")

model = DualFreqSleepStager(
    hf_input_channels=len(hf_features),
    lf_input_channels=len(lf_features),
    cnn_output_channels=16,
    cnn_hidden_channels=32,
    lstm_hidden_size=64,
    lstm_num_layers=2,
    lstm_bidirectional=True,
    dropout=0.1,
    num_sleep_stages=5,
    learning_rate=1e-3,
    weight_decay=5e-5,
    weight_tensor=None,
    convnet='CNN',
    debug=True
)

output = model(temp_hf.unsqueeze(0), temp_lf.unsqueeze(0))
print(f"output shape: {output.shape}")

model2 = DualFreqSleepStager(
    hf_input_channels=len(hf_features),
    lf_input_channels=len(lf_features),
    cnn_output_channels=16,
    cnn_hidden_channels=32,
    lstm_hidden_size=64,
    lstm_num_layers=2,
    lstm_bidirectional=True,
    dropout=0.1,
    num_sleep_stages=5,
    learning_rate=1e-3,
    weight_decay=5e-5,
    weight_tensor=None,
    convnet='TCN',
    debug=True
)
output2 = model2(temp_hf.unsqueeze(0), temp_lf.unsqueeze(0))
print(f"output2 shape: {output2.shape}")


# # Get Class Weights

# In[ ]:


# get class weights for weighted loss
all_labels = []
for batch in DataLoader(train_dataset, batch_size=1):
    labels = batch[2].numpy()
    all_labels.extend(labels.flatten())
all_labels = np.array(all_labels)
valid_labels = all_labels[all_labels != -1]
classes = np.arange(5)
class_counts = Counter(valid_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=valid_labels
)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")


# # Train Model

# In[40]:


cnn_output_channels = 64
lstm_hidden_size = 128
lstm_num_layers = 2
lstm_bidirectional = True
dropout = 0.2


wandb_logger = WandbLogger(
    project="mixed_freq_cnn_lstm",
    name="multiple-hf-channels"
)
checkpoint_callback = ModelCheckpoint(
    monitor='val_cohen_kappa',
    dirpath='checkpoints/mixed_freq_cnn_lstm/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='max'
)
early_stop_callback = EarlyStopping(
    monitor='val_cohen_kappa',
    patience=15,
    verbose=True,
    mode='max'
)
trainer = pl.Trainer(
    max_epochs=100,
    devices=1,
    accelerator='gpu',
    logger=wandb_logger,
    log_every_n_steps=1,
    precision="16-mixed",
    #callbacks=[checkpoint_callback, early_stop_callback]
    callbacks=[checkpoint_callback] # not using early stopping for now
)
model = DualFreqSleepStager(
    hf_input_channels=len(hf_features),
    lf_input_channels=len(lf_features),
    cnn_output_channels=cnn_output_channels,
    cnn_hidden_channels=32,
    lstm_hidden_size=lstm_hidden_size,
    lstm_num_layers=2,
    lstm_bidirectional=True,
    dropout=dropout,
    num_sleep_stages=5,
    learning_rate=1e-3,
    weight_decay=5e-5,
    weight_tensor=torch.tensor(class_weights, dtype=torch.float32),
    convnet='CNN',
    debug=False
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
trainer.fit(model, train_loader, val_loader)
wandb.finish()

