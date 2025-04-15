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
    def __init__(self, subjects_list, data_dir, chunk_duration=600, chunk_stride=300, downsample_freq=64, debug=False):
        """
        Args:
            subjects_list (list): List of subject IDs, e.g. ["SID1", "SID2", ...].
            data_dir (str): Directory where files like "SID_whole_df.csv" are stored.
            chunk_duration (int): Chunk length in seconds (default 600 s for 10 minutes).
            chunk_stride (int): Time in seconds to step forward between chunks (default 300 s, for 50% overlap).
            downsample_freq (int): Desired sampling frequency after downsampling (original data are at 64 Hz).
            debug (bool): If True, print status messages.
        """
        self.chunks = []  # List to store each generated chunk (with its corresponding data, labels, and SID)
        # Compute downsample factor (original sampling rate is 64 Hz)
        self.downsample = int(64 // downsample_freq)
        # Effective sampling rate after downsampling becomes downsample_freq Hz.
        self.chunk_length = int(chunk_duration * downsample_freq)
        self.stride = int(chunk_stride * downsample_freq)

        for SID in subjects_list:
            file_path = os.path.join(data_dir, f"{SID}_whole_df.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, dtype={'Sleep_Stage': 'category'}, converters=converters, low_memory=True)
                if debug:
                    print(f"Loaded data for subject {SID}")
                
                # Downsample: every self.downsample-th row
                if self.downsample != 1:
                    df = df.iloc[::self.downsample].reset_index(drop=True)
                    if debug:
                        print(f"After downsampling (factor {self.downsample}), rows: {len(df)}")
                
                # Remove rows with "Preparation" phase if labeled 'P'
                df = df[df['Sleep_Stage'] != 'P']

                # Ensure numeric conversion for required columns
                for col in ['ACC_X', 'ACC_Y', 'ACC_Z', 'BVP', 'TEMP', 'TIMESTAMP']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Compute accelerometer magnitude from three axes
                ACC = np.sqrt(df['ACC_X']**2 + df['ACC_Y']**2 + df['ACC_Z']**2)
                
                # Prepare the features: TIMESTAMP, BVP, TEMP and the computed ACC
                df_X = df[['TIMESTAMP', 'BVP', 'TEMP']].copy()
                df_X['ACC'] = ACC
                # Normalize the features (z-score normalization per subject)
                TEMP_norm = (df_X['TEMP'] - df_X['TEMP'].mean()) / df_X['TEMP'].std()
                df_X['TEMP'] = TEMP_norm
                BVP_norm = (df_X['BVP'] - df_X['BVP'].mean()) / df_X['BVP'].std()
                df_X['BVP'] = BVP_norm
                
                # Process sleep stage labels: trim whitespace and map to integer
                df['Sleep_Stage'] = df['Sleep_Stage'].astype(str).str.strip()
                df_Y = df['Sleep_Stage'].map(SLEEP_STAGE_MAPPING)
                
                # Convert features and labels to numpy arrays
                data_arr = df_X.values.astype(np.float32)  # shape: [T, C]
                labels_arr = df_Y.to_numpy()                # shape: [T]
                T = data_arr.shape[0]

                # If the record is too short (less than one chunk), pad it with NaNs (-1 for labels)
                if T < self.chunk_length:
                    pad_size = self.chunk_length - T
                    padding_data = np.full((pad_size, data_arr.shape[1]), np.nan, dtype=np.float32)
                    data_arr = np.concatenate([data_arr, padding_data], axis=0)
                    padding_labels = np.full((pad_size,), -1)
                    labels_arr = np.concatenate([labels_arr, padding_labels], axis=0)
                    T = self.chunk_length  # update length

                # Slide a window over the data with the defined stride to create overlapping chunks
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
        # Use forward_fill to replace any NaNs with previous values.
        data = forward_fill(data)
        labels = forward_fill(labels)
        return data, labels

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
                if len(df_X) > max_length:
                    if debug:
                        print(f"Truncating data for {SID} from {len(df_X)} to {max_length} samples.")
                    df_X = df_X.iloc[:max_length]
                    df_Y = df_Y.iloc[:max_length]
                else:
                    padding_length = max_length - len(df_X)
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
        labels = forward_fill(labels) # fill NaNs with previous values
        return data, labels

