"""
MNIST Logistic Regression: A script to classify digits 5 and 6 using logistic regression.
Includes data preprocessing, model training, evaluation, and hyperparameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score


# 1. Load and Preprocess the MNIST Dataset
def load_and_preprocess_data():
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target.astype(int)

    # Filter for digits 5 and 6 only
    filter_5_6 = (y == 5) | (y == 6)
    x, y = x[filter_5_6], y[filter_5_6]

    # Convert labels: 5 -> 0, 6 -> 1
    y = (y == 6).astype(int)

    # Normalize the data
    x = x / 255.0

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


# 2. Display Sample Images
def display_sample_images(x_train, y_train):
    plt.figure(figsize=(8, 4))
    for i, digit in enumerate([0, 1]):
        index = np.where(y_train == digit)[0][0]
        plt.subplot(1, 2, i + 1)
        plt.imshow(x_train[index].reshape(28, 28), cmap='gray')
        plt.title(f"Class {5 + digit}")
        plt.axis('off')
    plt.suptitle("Sample Images from Each Class (5 and 6)")
    plt.show()


# 3. Logistic Regression Models
def train_logistic_regression(x_train, y_train, c=1.0, max_iter=100, regularized=True):
    penalty = 'l2' if regularized else 'none'
    model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, solver='lbfgs')
    model.fit(x_train, y_train)
    return model


# 4. Evaluate Model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    return overall_accuracy, per_class_accuracy, conf_matrix


def print_evaluation_results(overall_acc, per_class_acc, model_name):
    print(f"\n{model_name}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print("Per-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {5 + i}: {acc:.4f}")


# 5. Train/Test Accuracy vs. Number of Samples
def plot_accuracy_vs_samples(x_train, y_train, x_test, y_test):
    sample_fractions = [0.1, 0.2, 0.5, 1.0]
    train_accuracies, test_accuracies = [], []

    for frac in sample_fractions:
        n_samples = int(len(x_train) * frac)
        x_train_subset, y_train_subset = x_train[:n_samples], y_train[:n_samples]
        model = train_logistic_regression(x_train_subset, y_train_subset)
        train_accuracies.append(model.score(x_train_subset, y_train_subset))
        test_accuracies.append(model.score(x_test, y_test))

    plt.figure(figsize=(8, 6))
    plt.plot(sample_fractions, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(sample_fractions, test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel("Fraction of Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy vs Fraction of Training Samples")
    plt.legend()
    plt.grid()
    plt.show()


# 6. Hyperparameter Tuning
def plot_accuracy_vs_regularization(x_train, y_train, x_test, y_test, c_values, max_iter):
    train_accuracies, test_accuracies = [], []

    for c in c_values:
        model = train_logistic_regression(x_train, y_train, c=c, max_iter=max_iter)
        train_accuracies.append(model.score(x_train, y_train))
        test_accuracies.append(model.score(x_test, y_test))

    plt.figure(figsize=(8, 6))
    plt.plot(c_values, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(c_values, test_accuracies, label='Test Accuracy', marker='s')
    plt.xscale('log')
    plt.xlabel("Regularization Strength (C)")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy vs Regularization Strength")
    plt.legend()
    plt.grid()
    plt.show()


# 7. Plot ROC Curve and AUROC
def plot_roc_curve_and_auroc(model, x_test, y_test, model_name):
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auroc = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUROC = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"{model_name} - AUROC: {auroc:.4f}")


# 8. Hyperparameter Tuning with GridSearchCV
def perform_grid_search(x_train, y_train, c_values, max_iter):
    param_grid = {'C': c_values}
    grid_search = GridSearchCV(
        LogisticRegression(penalty='l2', solver='lbfgs', max_iter=max_iter),
        param_grid, cv=3, scoring='accuracy', verbose=1
    )
    grid_search.fit(x_train, y_train)
    try:
        best_c = grid_search.best_params_['C']
        best_score = grid_search.best_score_
    except (KeyError, AttributeError):
        print("Grid search failed or parameter grid is empty. Returning default values.")
        best_c, best_score = None, None
    return best_c, best_score


# Main Execution
def main():
    # Load and preprocess the MNIST dataset
    x_train, x_test, y_train, y_test = load_and_preprocess_data()

    # Display sample images from each class (5 and 6)
    display_sample_images(x_train, y_train)

    # Train logistic regression models with and without regularization
    max_iter = 3000
    model_with_reg = train_logistic_regression(x_train, y_train, c=1.0, max_iter=max_iter)
    model_without_reg = train_logistic_regression(x_train, y_train, c=1e10, max_iter=max_iter, regularized=False)

    # Evaluate the performance of both models
    overall_acc_with_reg, per_class_acc_with_reg, _ = evaluate_model(model_with_reg, x_test, y_test)
    overall_acc_without_reg, per_class_acc_without_reg, _ = evaluate_model(model_without_reg, x_test, y_test)

    # Print evaluation results for both models
    print_evaluation_results(overall_acc_with_reg, per_class_acc_with_reg, "Logistic Regression with Regularization")
    print_evaluation_results(overall_acc_without_reg, per_class_acc_without_reg, "Logistic Regression without Regularization")

    # Analyze the effect of training sample size on accuracy
    plot_accuracy_vs_samples(x_train, y_train, x_test, y_test)

    # Analyze the effect of regularization strength on accuracy
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    plot_accuracy_vs_regularization(x_train, y_train, x_test, y_test, c_values, max_iter)

    # Plot ROC curves and calculate AUROC for both models
    plot_roc_curve_and_auroc(model_with_reg, x_test, y_test, "Logistic Regression with Regularization")
    plot_roc_curve_and_auroc(model_without_reg, x_test, y_test, "Logistic Regression without Regularization")

    # Perform hyperparameter tuning using GridSearchCV
    best_c, best_acc = perform_grid_search(x_train, y_train, c_values, max_iter)

    # Print the best hyperparameter values and corresponding accuracy
    print("\nFinal Hyperparameter Values (after Hyperparameter Tuning):")
    print(f"Best Regularization Strength (C): {best_c}")
    print(f"Best Cross-Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()