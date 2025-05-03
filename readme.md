# Deep Learning in Medicine - Group 6 Final Project
# Sleep Staging

This repository contains all code used in our project

## Project Structure

```
dl4med_final_project/
├── Lake/
│   ├── DualFreqSleepStager.ipynb         — Dual‐frequency dataset, TCN, LSTM & TCN‑LSTM experiments  
│   ├── run_jupyter_notebook.scr          — Slurm script for interactive notebook runs  
│   ├── run_notebook_nonint.sh            — Batch script for non-interactive runs  
│   └── __pycache__/                      
├── Tasha/
│   ├── ActiNetGRUModel.ipynb             — ActiNet‐GRU experiments  
│   ├── ActiNetLSTM.ipynb                 — ActiNet‐LSTM experiments  
│   ├── Baseline.ipynb                    — Simple baseline models  
│   ├── Baseline-GRU.ipynb                
│   ├── ManualGRUModel.ipynb              
│   ├── ManualLSTM.ipynb                  
│   └── wearables.py                      — Utility functions for wearable sensor data  
├── Mariam/
│   └── mariam_effnet_master.ipynb        — End‑to‑end preprocessing & 1D EfficientNet experiments  
├── Tanvi/
│   ├── sleep_staging_64hz.ipynb          — SleepPPG‑Net on 64 Hz BVP  
│   ├── sleep_staging_.2hz.ipynb          — SleepPPG‑Net on 0.2 Hz BVP  
│   ├── confusion_matrix_plotter.py       — Helper for plotting confusion matrices  
│   └── sleep_datasets.py                 — Dataset loaders for experiments  
└── README.md                             — This file  
```