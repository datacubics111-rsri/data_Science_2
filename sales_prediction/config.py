"""
Configuration file for Sales Prediction Project
Author: Data Science Team
Company: IBM
"""

import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Source
DATA_URL = 'https://www.kaggle.com/datasets/bumba5341/advertisingcsv'
RAW_DATA_FILE = RAW_DATA_DIR / 'advertising.csv'
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'advertising_processed.csv'

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature Engineering
TARGET_VARIABLE = 'Sales'
FEATURE_COLUMNS = ['TV', 'Radio', 'Newspaper']

# Visualization Settings
FIGURE_SIZE = (12, 6)
COLOR_PALETTE = 'viridis'
STYLE = 'whitegrid'

print("Configuration loaded successfully!")