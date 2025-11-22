import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
import pickle


# -----------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------
df = pd.read_csv("../data/bank_fin.csv", sep=";")


# -----------------------------------------------------------
# 2. Clean balance column “2 343,00 $” → 2343.00
# -----------------------------------------------------------
def clean_balance(x):
    if isinstance(x, str):
        x = x.replace(" ", "")
        x = x.replace("$", "")
        x = x.replace(",", ".")
    return float(x)

df["balance"] = df["balance"].apply(clean_balance)


# -----------------------------------------------------------
# 3. Target variable & feature set
# -----------------------------------------------------------
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

y = df["deposit"]
X = df.drop("deposit", axis=1)


# -----------------------------------------------------------
# 4. Numeric and Categorical features
# -----------------------------------------------------------
numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical_features = [col for col in X.columns if col not in numeric_features]


# -----------------------------------------------------------
# 5. Preprocessor (OneHotEncoder)
# -----------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# -----------------------------------------------------------
# 6. Optuna objective
# -----------------------------------------------------------
def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 100, 200)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    score = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()

    return score


# -----------------------------------------------------------
# 7. Run Optuna
# -----------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

print("Best parameters:", study.best_params)


# -----------------------------------------------------------
# 8. Train final model
# -----------------------------------------------------------
best_params = study.best_params

final_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=42
        )),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

final_model.fit(X_train, y_train)


# -----------------------------------------------------------
# 9. Evaluate
# -----------------------------------------------------------
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", round(acc, 2))


# -----------------------------------------------------------
# 10. Save model
# -----------------------------------------------------------
with open("../models/best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print("Model saved to ../models/best_model.pkl")
