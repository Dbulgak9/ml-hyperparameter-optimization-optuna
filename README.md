📦 ml-hyperparameter-optimization-optuna
A practical project demonstrating modern hyperparameter optimization with Optuna

This project demonstrates how to apply Optuna, a modern and efficient hyperparameter optimization framework, to tune a Random Forest Classifier for a supervised machine learning task.

It is designed as a clean, portfolio-ready example of a full ML workflow, including:

data preparation

model training

hyperparameter tuning

evaluation

model saving

🚀 Project Overview

The main goal is to show how hyperparameter optimization can significantly improve model performance compared to manually selected parameters or basic GridSearch.

The repository includes:

✔️ Reproducible Jupyter Notebook

✔️ Full Optuna optimization pipeline

✔️ Comparison with baseline model

✔️ Training script for automation

✔️ Saved optimized model (.pkl)

✔️ Clean and professional repo structure

📁 Repository Structure
ml-hyperparameter-optimization-optuna/

│
├── README.md                       # Documentation

├── .gitignore                      # Ignore rules for Python/ML

├── LICENSE                         # MIT License
│
├── notebooks/

│   └── 01_optuna_random_forest.ipynb   # Main notebook
│
├── src/

│   └── train_optuna_rf.py              # Script version of the pipeline
│
├── models/

│   └── best_random_forest.pkl          # Saved optimized model
│

└── data/

    └── (optional dataset files)

🧠 Optimization Details

Optuna is used to tune the following hyperparameters:

n_estimators (100 → 200)

max_depth (10 → 30)

min_samples_leaf (2 → 10)

The Optuna objective function uses 5-fold cross-validation and maximizes accuracy.

🔧 Technologies Used

Python 3.x

Scikit-Learn

Optuna

Pandas

NumPy

Matplotlib

Jupyter Notebook

📈 Results

Best model accuracy on the test dataset:

🟩 0.84

The optimized Random Forest model shows noticeably better performance than the baseline.

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt


If you don’t have a requirements file, install manually:

pip install optuna scikit-learn pandas numpy matplotlib

2. Run the notebook
jupyter notebook notebooks/01_optuna_random_forest.ipynb

3. Or run the script
python src/train_optuna_rf.py

💾 Loading the Saved Model
import pickle

with open("models/best_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

📊 Feature Importance (optional extension)

This project can be extended with:

SHAP analysis

permutation importance

classic feature_importances_

📜 License

This project is licensed under the MIT License — free for personal and commercial use.
