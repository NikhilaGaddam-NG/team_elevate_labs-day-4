Logistic Regression Classification

Objective

Build a binary classification model using Logistic Regression to predict outcomes based on input features.
This project demonstrates the complete pipeline from data loading to model evaluation, including threshold tuning and an explanation of the sigmoid function.

Tools and Libraries

- Python 3.x
- Pandas for data handling
- Scikit-learn for machine learning models and evaluation
- Matplotlib for visualization
- NumPy for mathematical operations

Steps Covered


1. Load a Binary Classification Dataset
- Used the Breast Cancer dataset from sklearn.datasets.
- Dataset includes features like mean radius, texture, perimeter, etc., and a target label indicating malignancy (0 = malignant, 1 = benign).

2. Train/Test Split and Standardization
- Split the dataset into 80% training and 20% testing sets.
- Standardized the features using StandardScaler to ensure all input features have mean 0 and variance 1.

3. Train a Logistic Regression Model
- Created a Logistic Regression model using Scikit-learn.
- Trained the model on the standardized training data.

4. Model Evaluation
- Predicted labels for the test data.
- Evaluated using:
  - Confusion Matrix
  - Precision Score
  - Recall Score
  - ROC-AUC Score
  - Classification Report
- Plotted the ROC Curve for visual evaluation of classifier performance.

5. Threshold Tuning
- By default, Logistic Regression uses a threshold of 0.5 for classification.
- The project explores setting a custom threshold of 0.3 to observe changes in Precision and Recall.
- Recomputed confusion matrix and performance metrics after threshold tuning.

6. Sigmoid Function Explanation
- Explained and plotted the Sigmoid function.
- Discussed how the sigmoid maps any real-valued number to a value between 0 and 1.
- Explained how the output probability from the sigmoid function is compared to a threshold to make final binary predictions.

Key Concepts Explained

- Confusion Matrix: Summarizes prediction results; shows True Positives, False Positives, True Negatives, and False Negatives.
- Precision: How many of the predicted positives are actually positive.
- Recall: How many actual positives were correctly predicted.
- ROC-AUC: Measures the ability of the model to distinguish between classes.
- Threshold Tuning: Adjusting the probability cutoff to balance precision and recall according to business needs.
- Sigmoid Function: Converts linear regression outputs into probabilities for classification.

How to Run

1. Clone the repository or download the script.
2. Install required libraries:
   pip install pandas scikit-learn matplotlib numpy
3. Run the Python script:
   python logistic_regression_classification.py
4. Outputs including metrics, graphs (ROC curve, Sigmoid curve), and explanations will be printed and plotted.

Sample Outputs

- Accuracy: High, since the Breast Cancer dataset is relatively clean and separable.
- Precision/Recall: Change depending on the threshold.
- ROC-AUC: Close to 1, indicating good model performance.

Future Enhancements

- Cross-validation for model validation.
- Hyperparameter tuning using GridSearchCV.
- Try other binary datasets (like Titanic, Bank Marketing datasets).
- Use advanced evaluation techniques like Precision-Recall Curves.
