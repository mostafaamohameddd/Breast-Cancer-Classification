## Breast Cancer Classification Project
## Overview
This project focuses on building a machine learning model to classify breast cancer tumors as Benign (non-cancerous) or Malignant (cancerous).
It uses the Breast Cancer Wisconsin Dataset, applies preprocessing and feature scaling, and trains a classifier to make predictions.
Additionally, a Flask web application is integrated to provide an easy-to-use interface for real-time predictions.

## Tech Stack
Programming Language: Python

Libraries:

Data Science → Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Machine Learning → Scikit-learn (for model training & evaluation)

Web Framework → Flask

Model Deployment: Flask app with HTML frontend

## Project Structure 
├── app.py                     # Flask web app for predictions
├── Machine_learning_project.ipynb   # Jupyter Notebook (EDA + training)
├── finalized_model.pkl        # Trained ML model (saved with pickle)
├── scaler.pkl                 # Scaler used for preprocessing
├── templates/
│   └── index.html             # Frontend for the web app
└── README.md                  # Project documentation

## How To Run
1️) Clone the repository
git clone https://github.com/mostafaamohameddd/breast-cancer-classification.git
cd breast-cancer-classification

2) Create & activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Linux/Mac

3) Install dependencies
pip install -r requirements.txt

4) Run the Flask app
python app.py

## Dataset

The dataset is from the Breast Cancer Wisconsin (Diagnostic) Dataset.
Features include tumor measurements such as radius, texture, smoothness, symmetry, etc.
Target variable:

M → Malignant

B → Benign

## Web App Preview

The Flask app provides a simple web form where users input tumor features, and the model outputs:

Malignant (cancerous)

Benign (non-cancerous)

## Results

Preprocessing: Feature scaling applied using StandardScaler.

Model: Trained ML classifier (details inside notebook).

Performance: High accuracy on test data.

