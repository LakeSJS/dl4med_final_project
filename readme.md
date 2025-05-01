# Deep Learning in Medicine - Group 6 Final Project
# Sleep Staging

This repository contains all code used in our project

## Project Structure

├── .gitignore   
├── Lake/  
│   ├── DualFreqSleepStager.ipynb   – main notebook implementing dual‐frequency dataset, TCN, LSTM, TCN‑LSTM  
│   ├── run_jupyter_notebook.scr  - script for running jupyter notebook on bigpurple
│   ├── run_notebook_nonint.sh  - sbatch script to run notebook on big purple non-interactively
│   └── __pycache__/  
├── Tasha/  
│   ├── ActiNetGRUModel.ipynb       – ActiNet GRU experiments  
│   ├── ActiNetLSTM.ipynb           – ActiNet LSTM experiments  
│   ├── Baseline.ipynb              – simple baselines  
│   ├── Baseline-GRU.ipynb  
│   ├── ManualGRUModel.ipynb  
│   ├── ManualLSTM.ipynb  
│   └── wearables.py                – utility functions for wearable sensor data  
