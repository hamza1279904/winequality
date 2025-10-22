

# 🍷 Wine Quality Prediction using Random Forest

This repository contains the code and resources for a machine learning project that predicts the quality of wine based on its physicochemical properties. The project utilizes a **Random Forest Classifier** to distinguish between "good" and "bad" quality wines.

## 📝 Overview

The primary goal of this project is to build a classification model that accurately predicts wine quality. The model is trained on a dataset of red wine samples, and the Random Forest algorithm was chosen for its effectiveness in handling complex data with multiple features. The analysis includes data exploration, preprocessing, model training, hyperparameter tuning, and a detailed performance evaluation.

-----

## 📂 Repository Structure

```
├── AssignmentpartC.ipynb
├── wine.csv
└── README.md
```

  * `AssignmentpartC.ipynb`: Jupyter Notebook containing the complete Python code for the analysis.
  * `wine.csv`: The dataset used for training and testing the model.
  * `README.md`: This file.

-----

## 💾 Dataset

The project uses the Wine Quality dataset containing 1599 rows and 12 columns.

  * **Features:** `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, and `alcohol`.
  * **Target Variable:** `quality`. This was converted into a binary variable where wines with a quality score \> 5 are labeled "good" (1) and those with a score ≤ 5 are labeled "bad" (0).

-----

## 🚀 Usage

All the code for data analysis, model training, and evaluation is in the Jupyter Notebook.

1.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
2.  Open `AssignmentpartC.ipynb`.
3.  Run the cells in the notebook to execute the code.

-----

## ⚙️ Project Workflow

The project follows a standard machine learning workflow:

1.  **Exploratory Data Analysis (EDA):** The dataset was loaded and checked for missing values; none were found. A correlation matrix revealed that **alcohol** has a positive correlation with quality, while **volatile acidity** has a negative correlation.

2.  **Data Preprocessing:** The categorical `quality` variable was label-encoded into a numeric format (0 for bad, 1 for good). Features were standardized using `StandardScaler`, which was fit only on the training data to prevent data leakage.

3.  **Model Training and Hyperparameter Tuning:** A Random Forest Classifier was trained to predict wine quality. `GridSearchCV` was used to find the best hyperparameters, which were determined to be `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}`.

4.  **Model Evaluation:** The model's performance was assessed on the unseen test set, using a confusion matrix and learning curves to understand its predictive behavior.

-----

## 📊 Results & Key Findings

  * **Performance Metrics:**

      * **Accuracy:** **72%**
      * **ROC-AUC Score:** **0.8234**

  * **Confusion Matrix Breakdown:**

      * **True Positives (Good as Good):** 62
      * **True Negatives (Bad as Bad):** 82
      * **False Positives (Bad as Good):** 25
      * **False Negatives (Good as Bad):** 31

  * **Insights:**

      * The learning curves indicated that the model was **overfitting**, as it achieved perfect scores on the training data but lower scores on the validation data.
      * Cross-validation analysis showed that using around **100 estimators** offers the best balance between model performance and computational efficiency.

-----

## 🌱 Future Improvements

While the model provides a solid baseline, the following steps could enhance its performance:

  * **Feature Engineering:** Create new interaction features to capture more complex, non-linear relationships in the data.
  * **Advanced Models:** Experiment with other powerful ensemble techniques like **Gradient Boosting**.
  * **Mitigate Overfitting:** Implement regularization techniques or gather more data to help the model generalize better.
  * **Feature Selection:** Use methods like Recursive Feature Elimination (RFE) to identify and keep only the most influential features, which can reduce noise and improve interpretability.

-----

## 👨‍💻 Author

  * **Hamza Jameel Hashmi**
  * **GitHub:** [https://github.com/hamza1279904](https://www.google.com/search?q=https://github.com/hamza1279904)
