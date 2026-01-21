# ML Hyperparameter Sensitivity Study

This repository contains my final project for the course **Principles of Machine Learning**.
The project focuses on studying **hyperparameter sensitivity** and **bias–variance trade-offs**
across multiple real-world classification datasets.

The objective of the project is not just to achieve high accuracy, but to understand how
different machine learning models behave when their hyperparameters are varied and when
they are applied to datasets with different characteristics.

---

## Datasets Used

The following datasets were used in this project:

- **Abalone Dataset** – Age classification based on physical measurements  
- **Credit Card Default Dataset** – Default payment prediction  
- **Mushroom Dataset** – Edible vs poisonous classification  
- **Student Dropout Dataset** – Academic success and dropout prediction  
- **Diabetes Dataset** – Binary classification for diabetes prediction  

Each dataset differs in size, feature space, and class distribution, making them suitable
for studying model stability and variance.

---

## Models Implemented

The following machine learning models were implemented and evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree

All models were tested under different hyperparameter configurations to study their
performance sensitivity.

---

## Project Structure

ML-hyperparameter-sensitivity-study/
- datasets/ # All datasets used in the project
- notebooks/ # Dataset-specific analysis notebooks
- framework/ # Reusable ML framework
  - models/ # Individual model implementations
  - preprocessing.py
  - sensitivity_analysis.py
  - main.py
- results/ # Result summaries
- report/ # Final report and presentation
- requirements.txt # Project dependencies
- README.md

---

## Note on Diabetes Dataset

The diabetes dataset is used within the **custom machine learning framework**
to perform automated hyperparameter sensitivity and variance analysis.
Unlike other datasets, it is not explored using a standalone Jupyter notebook.

---

## Key Concepts Explored

- Hyperparameter sensitivity analysis  
- Bias–variance trade-off  
- Model stability across datasets  
- Impact of dataset characteristics on model performance  

---

## HOW TO RUN THE PROJECT

1. Clone the repository:
git clone https://github.com/shanaysheth/ML-hyperparameter-sensitivity-study.git

2. Navigate to the project directory:
cd ML-hyperparameter-sensitivity-study

3. Install dependencies:
pip install -r requirements.txt

4. Run the framework:
python framework/main.py

---

## Academic Context:
This project was completed as part of an academic course.
Each student was assigned different datasets; this repository reflects my
independent implementation, analysis, and interpretation.


## Author
Shanay Sheth
Undergraduate Student – Computer Science / Business Analytics
