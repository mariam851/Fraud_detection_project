# Setup Instructions for Running the Project

This project requires the dataset and trained model to work locally.

## 1. Dataset
Download the dataset from Kaggle:

https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

Place the CSV file in a folder named `data` at the root of the project:

project_root/
└─ data/
└─ fraud_dataset.csv


## 2. Trained Model
The trained pipeline (`fraud_detection_pipeline.pkl`) is not included due to file size. 
You can generate it by training the model locally:

```bash
python train_model.py

```python
# train_model.py
from fraud_app import train_pipeline

if __name__ == "__main__":
    data_path = "data/fraud_dataset.csv"
    model_path = "models/fraud_detection_pipeline.pkl"
    
    train_pipeline(data_path, model_path)
    print(f"Trained model saved at {model_path}")

