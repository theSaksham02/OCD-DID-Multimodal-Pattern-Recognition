#!/bin/bash

echo "=== Downloading Datasets for OCD-DID Pattern Recognition ==="

# Create data directories
mkdir -p data/raw/ocd_clinical
mkdir -p data/raw/fer2013
mkdir -p data/raw/pose_data

# Download OCD Clinical Dataset
echo "Downloading OCD Clinical Dataset..."
kaggle datasets download -d ohinhaque/ocd-patient-dataset-demographics-and-clinical-data -p data/raw/ocd_clinical --unzip

# Download FER2013
echo "Downloading FER2013 Dataset..."
kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013 --unzip

# Download alternative OCD dataset
echo "Downloading Additional OCD Data..."
kaggle datasets download -d wijdanalmutairi/ocd-dataset -p data/raw/ocd_clinical --unzip

echo "=== Download Complete! ==="
echo "Datasets saved in data/raw/"
