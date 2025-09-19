<artifact artifact_id="breast-cancer-prediction-readme" artifact_version_id="2" title="README.md" contentType="text/markdown">
# Breast Cancer Diagnosis Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightblue)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit-learn-1.2%2B-green)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web-based machine learning application for predicting breast cancer diagnosis (malignant or benign) using features from the Wisconsin Breast Cancer Dataset. Built with Flask for deployment and Logistic Regression for classification, this project demonstrates end-to-end ML workflow from data preprocessing to interactive prediction.

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **Interactive Web Interface**: User-friendly Flask app to input tumor features and get instant predictions.
- **Logistic Regression Classifier**: Trained on standardized features for high-accuracy binary classification.
- **Feature Scaling**: Uses StandardScaler to normalize inputs for optimal model performance.
- **Selected Features**: Focuses on 13 key morphological features (e.g., radius_mean, texture_mean) for concise input.
- **Error Handling**: Validates numerical inputs and provides clear feedback.
- **Model Persistence**: Saves and loads trained model/scaler via pickle for seamless deployment.
- **Extensible**: Easily adaptable for additional ML models or dataset expansions.

## Demo
The web app presents a form for entering 13 tumor features. Upon submission:

- **Input Example**: radius_mean=17.99, texture_mean=10.38, etc.
- **Output**: "Prediction: Malignant (cancerous)" or "Benign (non-cancerous)" displayed on the page.

![Demo Screenshot](demo_screenshot.png) *(Add a screenshot of the app in action here)*

## Prerequisites
- Python 3.8 or higher
- Access to Jupyter/Colab for model training (optional, pre-trained model included)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/mostafaamohameddd/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install flask scikit-learn pandas numpy matplotlib seaborn
   ```

**Note**: The pre-trained model (`finalized_model.pkl`) and scaler (`scaler.pkl`) are included. For retraining, see [Training the Model](#training-the-model). The dataset (`breast-cancer.csv`) is required for training—download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Usage
1. Run the Flask app:
   ```
   python app.py
   ```
   The app starts at `http://127.0.0.1:5000/`.

2. Open the URL in your browser, enter the 13 feature values in the form, and submit.
3. View the prediction result on the page.

For command-line prediction (from notebook), run the Jupyter cells for interactive input.

## Training the Model
The model is trained on the Wisconsin Breast Cancer Dataset using Logistic Regression.

1. Open `Machine_learning_project.ipynb` in Jupyter or Google Colab.
2. Run the cells sequentially:
   - Load and explore the dataset (`breast-cancer.csv`).
   - Preprocess: Handle missing values, encode diagnosis ('M'/'B'), select features.
   - Split data (train/test), scale features with StandardScaler.
   - Train Logistic Regression classifier.
   - Evaluate: Accuracy score, visualizations (e.g., correlation heatmap).
   - Save model and scaler as pickle files.
3. Key Metrics: Achieves high accuracy (~95-98% on test set, depending on splits).

**Feature Selection**: Uses 13 core features for simplicity and performance:
- radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean
- radius_se, texture_se, smoothness_se, compactness_se, concave points_se, symmetry_se
- symmetry_worst

Adapt for other classifiers (e.g., SVM, Random Forest) by modifying the notebook.

## Project Structure
```
breast-cancer-prediction/
├── app.py                  # Flask web application for predictions
├── Machine_learning_project.ipynb  # Jupyter notebook for training and evaluation
├── finalized_model.pkl     # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler for feature normalization
├── breast-cancer.csv       # Dataset (add to .gitignore for large files)
├── templates/
│   └── index.html          # HTML template for the web form
├── requirements.txt        # Python dependencies (generate with pip freeze)
└── README.md               # This file
```

**Note**: Create a `templates` folder with `index.html` for the Flask app (simple form for the 13 features).

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Report issues or suggest improvements via GitHub Issues.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (create one if not present).

## Acknowledgments
- UCI Machine Learning Repository for the Breast Cancer Wisconsin dataset.
- Scikit-learn for robust ML tools.
- Flask for lightweight web deployment.
- Inspired by healthcare AI projects on GitHub.

---


</artifact>
