# End-to-End Credit Risk & Default Prediction Pipeline

An enterprise-grade, production-ready Machine Learning pipeline designed to predict credit risk and loan default probabilities using advanced gradient boosting techniques. This project transitions from an exploratory data analysis phase into a modular, object-oriented software architecture tailored for FinTech applications.

## 🏗️ Project Architecture & Design Pattern
The system is built using Object-Oriented Programming (OOP) principles, ensuring high modularity, maintainability, and scalability for live banking systems:

* `src/data_preprocessing.py`: Contains the `CreditDataPreprocessor` class responsible for data ingestion, handling missing values via median imputation, and performing One-Hot Encoding with dummy variable trap avoidance.
* `src/model_training.py`: Contains the `CreditModelTrainer` class that handles stratified train-test splitting and orchestrates the model training process.
* `main.py`: The central execution gateway (orchestrator) that runs the entire production pipeline with a single command.

---

## 📊 Dataset & Business Logic Cleaning
The pipeline automatically handles missing values and drops critical data anomalies based on real-world banking constraints:
* **Age Filter:** Removes records where `person_age >= 100` (biologically impossible).
* **Employment Filter:** Removes records where `person_emp_length >= 60` (logically impossible).
* **Imputation Strategy:** Fills missing values in interest rates (`loan_int_rate`) and employment history using the **median** to prevent data skewness.

---

## 🤖 Machine Learning Model & Performance
We utilized the **XGBoost (Extreme Gradient Boosting)** classifier, which is the industry standard for tabular credit scoring data due to its robustness against non-linear financial patterns.

### Evaluation Metrics (Test Set Results):
* **Overall Accuracy:** `94%`
* **Non-Default (Class 0) Recall:** `99%`
* **Default / High-Risk (Class 1) Precision:** `97%`
* **Default / High-Risk (Class 1) Recall:** `73%`

> **Note on Risk Mitigation:** White the model exhibits an exceptional `97%` Precision for high-risk customers (minimizing false rejections of good clients), future iterations will focus on threshold optimization to push the `73%` Recall higher, minimizing financial loss from undetected bad loans.

---

## 🚀 How to Run the Production Pipeline

### 1. Installation
Ensure you have Python 3.11+ and the required packages installed:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn


🛠️ Tech Stack
Language: Python

Frameworks: XGBoost, Scikit-Learn

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Version Control: Git / GitHub