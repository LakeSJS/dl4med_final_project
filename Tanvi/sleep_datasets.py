from glob import glob
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from scipy.signal import cheby2, filtfilt
import numpy as np
import torch 

SLEEP_STAGE_MAPPING = {
    "W": 0,    # Wake
    "N1": 1,   # non-REM stage 1 (light)
    "N2": 1,   # non-REM stage 2 (light)
    "N3": 2,   # non-REM stage 3 (deep)
    "R": 3,    # REM
    "Missing": -1  # Missing label
}


class SleepDataset(Dataset):
    def __init__(self, subjects_list, data_dir, max_length, downsample_freq=64, debug=False):
        self.subjects = subjects_list
        self.data_dir = data_dir
        self.max_length = max_length
        self.downsample = int(64//downsample_freq)

        #make a pandas series of the file paths in the directory containing the subject id's of interest
        files = []
        for i in glob(self.data_dir + f"/*.csv"):
            for j in self.subjects:
                if j in i:
                    files.append(i)
        files = pd.Series(files)

        #define a function that reads in a file, performs the preprocessing steps, and returns the sequences/labels
        def get_sequences_and_labels(x):
            #read in the csv file of interest
            SID = x.split(self.data_dir+"/")[-1].split("_")[0]
            df = pd.read_csv(x,dtype={'TIMESTAMP': np.float64, 'BVP': np.float64, 'ACC_X': np.float64, 'ACC_Y': np.float64,
                                      'ACC_Z': np.float64, 'TEMP': np.float64, 'EDA': np.float64, 'HR': np.float64,
                                      'IBI':np.float64, 'Sleep_Stage': str})
            df = df.loc[df['Sleep_Stage'] != 'P'] # remove data before PSG start
            df_Y = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING) #map sleep stage labels to tokens
            
            bvp = df['BVP'].to_numpy()
            df["Class"] = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING).to_numpy()
            
            #implement low pass filter over BVP and make column BVP_FILT
            fs = 64  # Sampling frequency (Hz), adjust as needed
            nyq = 0.5 * fs  # Nyquist frequency
            cutoff = 8  # Desired cutoff frequency (Hz)
            order = 8 #order of chebyshev filter
            stopband_atten = 40  # Stop-band attenuation in dB
            wn = cutoff / nyq  # Normalize the cutoff frequency
            b, a = cheby2(order, stopband_atten, wn, btype='low', analog=False)
            bvp_filtered = filtfilt(b, a, bvp) ##SHAPE = (2304000,)
            df["BVP_FILT"] = bvp_filtered

            #downsample the whole dataframe to new desired frequency
            df_downsampled = df.copy(deep=True)
            if self.downsample != 1:
                df_downsampled = df_downsampled.iloc[::self.downsample]
    
            #standardize and clean the resampled BVP_FILT and make column BVP_NORM
            sd = df_downsampled.BVP_FILT.std()
            u = df_downsampled.BVP_FILT.mean()
            df_downsampled.loc[:,"BVP_NORM"] = (df_downsampled.BVP_FILT - u)/sd ##SHAPE = (1228681,)
            
            #extract the processed sequence and labels, and pad or truncate to desired length
            sequences = df_downsampled["BVP_NORM"].values 
            labels = df_downsampled["Class"].values            
            adj_length = int(self.max_length - len(sequences))#(df.TIMESTAMP.max() - df.TIMESTAMP.min())*fs - 1) #how much to adjust the sequence length by
                    
            if adj_length > 0:
                seq_padding = np.ones((adj_length))*0 #pad 0.0 for bvp values
                lab_padding = np.ones((adj_length))*-1 #pad -1 for "missing" sleep stage values
            
                sequences_padded = np.concat([sequences, seq_padding])
                labels_padded = np.concat([labels, lab_padding])
            elif adj_length < 0: #clip to 10 hours
                sequences_padded = sequences[:adj_length] 
                labels_padded = labels[:adj_length]
            else:
                sequences_padded = sequences ##SHAPE = (2304000,)
                labels_padded = labels ##SHAPE = (2304000,)
            
            #cast sequence and labels to tensor and return as a series
            sequences = torch.tensor(sequences_padded, dtype=torch.float32).unsqueeze(0)
            labels = torch.tensor(labels_padded,  dtype=torch.long)
            
            return pd.Series({"sequence": sequences, "labels": labels, "SID": SID})
        
        #apply the function to preprocess and get sequences/labels to all desired files, result is a dataframe with index "SID" and columns sequence, labels
        self.sequences_labels = files.apply(lambda x: get_sequences_and_labels(x)).set_index("SID")

    def __len__(self):
            return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects.iloc[idx]
        sequence = self.sequences_labels.loc[sid,"sequence"]
        label = self.sequences_labels.loc[sid,"labels"]
        return sequence, label

class SleepChunkDataset(Dataset):
    def __init__(self, subjects_list, data_dir, max_length, chunk_duration=600, chunk_stride=300, downsample_freq=32):
        self.subjects = subjects_list
        self.data_dir = data_dir
        self.max_length = max_length
        self.downsample = int(64//downsample_freq)
        self.chunks = []
        self.chunk_length = int(chunk_duration * downsample_freq)
        self.stride = int(chunk_stride * downsample_freq)       
        
        #make a pandas series of the file paths in the directory containing the subject id's of interest
        files = []
        for i in glob(self.data_dir + f"/*.csv"):
            for j in self.subjects:
                if j in i:
                    files.append(i)
        files = pd.Series(files)
        
        #define a function that reads in a file, performs the preprocessing steps, and returns the chunked sequences/labels
        def get_sequences_and_labels(x):
            #read in csv of interest as a dataframe
            SID = x.split(self.data_dir+"/")[-1].split("_")[0]
            df = pd.read_csv(x,dtype={'TIMESTAMP': np.float64, 'BVP': np.float64, 'ACC_X': np.float64, 'ACC_Y': np.float64, \
                                                                    'ACC_Z': np.float64, 'TEMP': np.float64, 'EDA': np.float64, 'HR': np.float64,
                                      'IBI':np.float64, 'Sleep_Stage': str})
            df = df.loc[df['Sleep_Stage'] != 'P'] # remove data before PSG start
            df_Y = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING) #map sleep stage labels to tokens
            
            bvp = df['BVP'].to_numpy()
            df["Class"] = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING).to_numpy()
            
            #implement low pass filter over BVP column and make new column BVP_FILT
            fs = 64  # Sampling frequency (Hz), adjust as needed
            nyq = 0.5 * fs  # Nyquist frequency
            cutoff = 8  # Desired cutoff frequency (Hz)
            order = 8 #order of chebyshev filter
            stopband_atten = 40  # Stop-band attenuation in dB
            wn = cutoff / nyq  # Normalize the cutoff frequency
            b, a = cheby2(order, stopband_atten, wn, btype='low', analog=False)
            bvp_filtered = filtfilt(b, a, bvp) ##SHAPE = (2304000,)
            df["BVP_FILT"] = bvp_filtered

            #downsample whole dataframe to desired frequency
            if self.downsample != 1:
                df_downsampled = df.copy(deep=True).iloc[::self.downsample]
            
            #standardize and clean BVP_FILT and make new column BVP_NORM
            sd = df_downsampled.BVP_FILT.std()
            u = df_downsampled.BVP_FILT.mean()
            df_downsampled.loc[:,"BVP_NORM"] = (df_downsampled.BVP_FILT - u)/sd ##SHAPE = (1228681,)
            
            #split the downsampled sequence/labels into windows (chunks) whose length and count are defined by chunk_duration, stride, downsample_freq, etc.
            chunks = []
            T = df_downsampled.shape[0]
            for start in range(0, T - self.chunk_length  + 1, self.stride):
                    end = start + self.chunk_length  - 1 
                    chunked_df = df_downsampled.loc[start:end, ["BVP_NORM","Class"]]
                    chunks.append(chunked_df)
            return chunks

        #get the chunks containing the windowed sequence/labels for all desired files/subjects and append them to the list self.allchunks
        file_chunks = files.apply(lambda x: get_sequences_and_labels(x))#.set_index("SID")
        self.allchunks = []
        for i in range(len(file_chunks)):
            self.allchunks += file_chunks[i]

    def __len__(self):
            return len(self.allchunks)

    def __getitem__(self, idx):
        chunk = self.allchunks[idx]
        sequence = torch.tensor(chunk["BVP_NORM"].values, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(chunk["Class"].values, dtype=torch.long)
        return sequence, label