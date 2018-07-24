#!/usr/bin/env python3

import os
import zipfile
import pandas as pd

# Module constants
DATASET_DIR = os.path.join(os.path.abspath(__file__), 'datasets')
TRAINING_CSV =  os.path.join(DATASET_DIR, 'train.csv')
TEST_CSV = os.path.join(DATASET_DIR, 'test.csv')
DATA_ZIP = os.path.join(DATASET_DIR, 'all.zip')


# get dataset if flat file not readily available
def get_csv_from_zip(train_set=True):
    csv = TRAINING_CSV if train_set else TEST_CSV

    with zipfile.ZipFile(DATA_ZIP) as data_zip:
        member = 'train.csv' if train_set else 'test.csv'
        data_zip.extract(member, path=DATASET_DIR)


# load data via pandas
def load_dataset(csv=TRAINING_CSV):
    if not os.path.isfile(csv):
        get_csv_from_zip()

    return pd.read_csv(csv)

############################################
# Explore the data
############################################
train_set = load_dataset()
