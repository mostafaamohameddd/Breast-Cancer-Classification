Breast Cancer Classification Application

A web-based machine learning application designed for classifying breast cancer diagnosis (malignant or benign) using features from the Wisconsin Breast Cancer Dataset. Powered by Flask for a seamless user interface and Logistic Regression for accurate classification, this project showcases a complete ML pipeline from data preprocessing to real-time prediction. Last updated: 02:26 PM EEST, Friday, September 19, 2025.
Table of Contents

Features
Demo
Prerequisites
Installation
Usage
Training the Model
Project Structure
Contributing
License
Acknowledgments

Features

Interactive Web Interface: User-friendly Flask app to input 13 tumor features and get instant predictions.
Logistic Regression Classifier: Trained on standardized features with ~95-98% accuracy on test data.
Feature Scaling: Uses StandardScaler for normalized inputs, ensuring robust model performance.
Input Validation: Checks for missing values and enforces reasonable ranges for all features.
Error Handling: Provides clear feedback for invalid inputs with detailed error messages.
Model Persistence: Saves and loads trained model/scaler via pickle for seamless deployment.
Extensible: Easily adaptable for additional ML models or dataset enhancements.

Demo
The web app presents a form for entering 13 tumor features (e.g., radius_mean, texture_mean). Upon submission:

Input Example: radius_mean=17.99, texture_mean=10.38, etc.
Output: "Prediction: Malignant (cancerous)" or "Benign (non-cancerous)" displayed on the page.

 
Prerequisites

Python 3.8 or higher
Access to Jupyter/Colab for model training (optional, pre-trained model included)

Installation

Clone the repository:
git clone https://github.com/mostafaamohameddd/breast-cancer-classification.git
cd breast-cancer-classification


Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install flask scikit-learn pandas numpy matplotlib seaborn



Note: The pre-trained model (finalized_model.pkl) and scaler (scaler.pkl) are included. For retraining, see Training the Model. The dataset (breast-cancer.csv) is required for training—download from UCI ML Repository.
Usage

Run the Flask app:
python app.py

The app starts at http://127.0.0.1:5000/.

Open the URL in your browser, enter the 13 feature values in the form, and submit.

View the prediction result on the page.


For command-line prediction, run the Jupyter notebook cells for interactive input.
Training the Model
The model is trained on the Wisconsin Breast Cancer Dataset using Logistic Regression.

Open Machine_learning_project.ipynb in Jupyter or Google Colab.
Run the cells sequentially:
Load and preprocess the dataset (breast-cancer.csv), handling missing values.
Select 13 key features and encode diagnosis ('M'/'B' as 1/0).
Split data (80% train, 20% test), scale features with StandardScaler.
Train Logistic Regression with cross-validation (5-fold).
Evaluate: Accuracy score, classification report.
Save model and scaler as pickle files.


Key Metrics: Achieves ~95-98% accuracy on test data (varies with splits).

Feature List:

radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean
radius_se, texture_se, smoothness_se, compactness_se, concave points_se, symmetry_se
symmetry_worst

Adapt for other classifiers (e.g., SVM, Random Forest) by modifying the notebook.
Project Structure
breast-cancer-classification/
├── app.py                  # Flask web application for predictions
├── Machine_learning_project.ipynb  # Jupyter notebook for training and evaluation
├── finalized_model.pkl     # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler for feature normalization
├── breast-cancer.csv       # Dataset (add to .gitignore for large files)
├── templates/
│   └── index.html          # HTML template for the web form
├── requirements.txt        # Python dependencies (generate with pip freeze)
└── README.md               # This file

Note: Create a templates folder with index.html (a simple form for the 13 features).
Contributing
Contributions are welcome! Please:

This project is licensed under the MIT License - see the LICENSE file for details (create one if not present).
Acknowledgments

UCI Machine Learning Repository for the Breast Cancer Wisconsin dataset.
Scikit-learn for robust ML tools.
Flask for lightweight web deployment.
Inspired by healthcare AI projects on GitHub.

