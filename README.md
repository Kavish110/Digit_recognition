# Logistic Regression on MNIST (Digits 5 vs 6)

This project applies logistic regression to classify digits 5 and 6 from the MNIST dataset. It demonstrates:

- Data preprocessing
- Model training (with and without regularization)
- Evaluation with accuracy metrics and confusion matrices
- ROC curve plotting
- Hyperparameter tuning using `GridSearchCV`

## 🧠 Model Description

Two logistic regression models are trained:

- **With L2 regularization**
- **Without regularization**

The models are evaluated based on overall accuracy, per-class accuracy, and AUROC scores.

## 📊 Visualization

- Sample digits display (5 and 6)
- Accuracy vs. training sample size
- Accuracy vs. regularization strength (C)
- ROC curves

## 🧪 Requirements

Install the required libraries using:

```bash
pip install numpy matplotlib scikit-learn
