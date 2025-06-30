# Credit Card Fraud Detection

This project demonstrates a basic machine learning workflow for detecting credit card fraud using logistic regression. It includes data extraction, preprocessing, feature engineering, model training, and evaluation.

## Project Structure

- `main.py`: Main script for data processing, model training, and evaluation.
- `archive (1).zip`: Zip file containing the dataset (`creditcard.csv`).
- `extracted/creditcard.csv`: Extracted CSV data file.
- `Copy_of_creditCardFraudDetector.ipynb`: Jupyter notebook with exploratory analysis and experiments.

## How to Run

1. Ensure you have Python 3 and the required libraries:
    - pandas
    - scikit-learn

2. Place `archive (1).zip` in the project directory.

3. Run the main script:
    ```sh
    python main.py
    ```

## Dataset

The dataset is from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Project Steps

- Data extraction from zip archive
- Data exploration and preprocessing
- Feature engineering (e.g., extracting hour from transaction time)
- Model training (Logistic Regression)
- Model evaluation (confusion matrix, classification report, AUC-ROC)

## Notes

- This script is for educational purposes and demonstrates a simple workflow.
- For production use, further improvements such as advanced preprocessing, model tuning, deployment, and monitoring are recommended.
