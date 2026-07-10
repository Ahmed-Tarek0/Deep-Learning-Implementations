# Diabetes Prediction — SVM Classifier

A machine learning project that predicts whether a patient has diabetes based on diagnostic medical measurements, using a Support Vector Machine (SVM) classifier with hyperparameter tuning via GridSearchCV.

## Dataset

The [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (`diabetes.csv`), containing 768 records.

## Project Workflow

### 1. Data Cleaning
- Identified biologically invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, and converted them to `NaN`.
- Found that **~49% of `Insulin`** values were missing — too significant to safely impute — so the column was **dropped**.
- Filled remaining missing values in `Glucose`, `BloodPressure`, `SkinThickness`, and `BMI` with the **median**.

### 2. Exploratory Data Analysis (EDA)
- Visualized the class distribution of the `Outcome` variable (bar chart).
- Compared feature means grouped by `Outcome` to spot differences between diabetic and non-diabetic patients.
- Plotted a correlation heatmap across all features.

### 3. Preprocessing
- Split features (`x`) and target (`y`), with `Outcome` as the label.
- Train/test split: 80/20 (`random_state=42`) → 614 training samples, 154 test samples.
- Standardized features using `StandardScaler`.

### 4. Modeling
- **Baseline model:** `SVC()` with default parameters.
- **Tuned model:** `GridSearchCV` over an SVM with `class_weight='balanced'`, searching:
  - `rbf` kernel: `C ∈ [0.1, 1, 10, 100]`, `gamma ∈ [1, 0.1, 0.01, 0.001]`
  - `linear` kernel: `C ∈ [0.1, 1, 10, 100]`
  - 5-fold cross-validation, optimized for **F1-score**

### 5. Predictive System
A simple inference example that takes a new patient's raw measurements, scales them with the fitted `scaler`, and predicts the diabetes outcome using the best tuned model.

## Results

| Model | Test Accuracy | Recall (Diabetic) | False Negatives |
|---|---|---|---|
| Baseline SVM | 76% | 62% | 21 |
| Tuned SVM (GridSearchCV) | 74% | 80% | 11 |

**Best hyperparameters:** `{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}`

**Key takeaway:** although the baseline model scored slightly higher on overall accuracy, the tuned model — using class balancing and F1-optimized grid search — substantially improved recall for diabetic patients (62% → 80%), cutting false negatives nearly in half (21 → 11). For a medical screening task, minimizing missed diabetic cases is more important than raw accuracy, making the tuned model the more suitable choice.

## Tech Stack

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (`SVC`, `GridSearchCV`, `StandardScaler`, `Pipeline`)

## Notes

- This project uses the classic "practice via YouTube tutorial" approach for reinforcing ML fundamentals (SVM classification, GridSearchCV, and handling class imbalance in a medical dataset context).

## Key Learnings

- SVM is highly sensitive to feature scaling
- Medical datasets require careful handling of missing values and outliers
- Accuracy alone can be misleading in classification problems
- Recall is especially important in healthcare applications to reduce false negatives
- `GridSearchCV` helps improve model performance through hyperparameter tuning
- Class balancing can improve recall at the cost of lower precision or accuracy
