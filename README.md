# Fashion-MNIST ANN Pipeline with DVC and Git

## Project Purpose
This repository demonstrates an end-to-end Machine Learning Operations (MLOps) workflow using Git for source code versioning, and DVC (Data Version Control) for tracking data, models, and ML pipelines. It was built for the FAST NUCES MLOps Assignment 3.

## Dataset
We use the **Fashion-MNIST** dataset. It consists of 60,000 training images and 10,000 testing images of clothing items (28x28 grayscale).

## Model Architecture
The model is a fully-connected Artificial Neural Network (ANN) consisting of:
- Flatten Layer (28x28)
- Dense Layer (ReLU)
- Dropout Layer
- Dense Output Layer (10 classes, Softmax)

## Project Structure
- `src/prepare.py`: Downloads/loads the raw data.
- `src/preprocess.py`: Normalizes and splits the data.
- `src/train.py`: Trains the ANN model.
- `src/evaluate.py`: Evaluates the model, generates a confusion matrix and metrics.
- `params.yaml`: Centralized hyperparameters.
- `dvc.yaml`: DVC pipeline definitions.

## Environment Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.\.venv\Scripts\activate.ps1`
3. Install dependencies: `pip install tensorflow "dvc[gdrive]" pyyaml scikit-learn matplotlib`

## Running the Pipeline
To run the fully automated ML pipeline:
```bash
dvc repro
```
To push the artifacts to the Google Drive remote:
```bash
dvc push
```
