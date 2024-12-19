Credit Card Fraud Detection

Using Machine Learning to Identify Fraudulent Transactions.

Project Overview

Credit card fraud detection is a critical issue in the financial sector. This project leverages machine learning techniques to accurately detect fraudulent transactions in a highly imbalanced dataset.

The solution uses a Random Forest Classifier, trained on a dataset of anonymized credit card transactions, with additional tools for data preprocessing, exploration, and performance evaluation.

Key Features

	•	Handles imbalanced data using under-sampling techniques.
	•	Utilizes Random Forest Classifier for robust predictions.
	•	Provides performance evaluation using ROC-AUC, Precision, Recall, and more.
	•	Visualizes key insights using EDA (Exploratory Data Analysis).

Table of Contents

	1.	Dataset
	2.	Requirements
	3.	Installation
	4.	How to Run
	5.	Project Highlights
	6.	Performance Metrics
	7.	Future Enhancements
	8.	Contributors

Dataset

The dataset used in this project is publicly available on Kaggle.
	•	Source: Credit Card Fraud Detection Dataset
	•	Details:
	•	284,807 transactions over two days.
	•	492 fraudulent cases (0.172% of transactions).
	•	Features: 28 anonymized PCA components (V1 to V28), Time, and Amount.

Instructions to Download the Dataset

	1.	Go to the dataset page on Kaggle: Credit Card Fraud Detection Dataset.
	2.	Sign in or create a Kaggle account if you don’t have one.
	3.	Click on the Download button to get the creditcard.csv file.
	4.	Place the file in the project directory.

Requirements

The project is built in Python and requires the following dependencies:

pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0.1
matplotlib==3.4.3
seaborn==0.11.2
joblib==1.1.0

Installation

	1.	Clone the repository:

git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection


	2.	Install the dependencies:

pip install -r requirements.txt

How to Run

Using Jupyter Notebook

	1.	Place the downloaded creditcard.csv in the project directory.
	2.	Open the notebook:

jupyter notebook "Credit Card Fraud Detection.ipynb"


	3.	Run all the cells to preprocess data, train the model, and evaluate its performance.

Project Highlights

	•	Preprocessing:
	•	Normalized Time and Amount for consistent scaling.
	•	Handled data imbalance using under-sampling.
	•	EDA:
	•	Visualized transaction patterns and class distributions.
	•	Explored feature correlations and fraud characteristics.
	•	Model Training:
	•	Built a Random Forest Classifier achieving 93% accuracy and 98% ROC-AUC.
	•	Evaluation:
	•	Evaluated performance using:
	•	Confusion Matrix
	•	Precision, Recall, and F1-Score
	•	ROC-AUC Score

Performance Metrics

Metric	Value
Accuracy	93%
Precision (Fraud)	96%
Recall (Fraud)	90%
F1-Score (Fraud)	93%
ROC-AUC Score	98%

Future Enhancements

	•	Add LightGBM and XGBoost models for comparison.
	•	Implement hyperparameter tuning to improve model performance.
	•	Deploy on a cloud platform like AWS, Azure, or Google Cloud.

