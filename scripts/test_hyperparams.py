import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Train several classification models on the training data and return a
    DataFrame summarising the best hyperparameters found for each model.

    Models trained:
        - Logistic Regression (baseline, default params)
        - Random Forest (RandomizedSearchCV)
        - Decision Tree (RandomizedSearchCV)
        - KNN (RandomizedSearchCV)
        - Gaussian Naive Bayes (GridSearchCV)

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per model containing the model name and the
        best hyperparameters as a list in the 'best_params' column.
    """
    results = []

    # ---------------------------------------------------------------
    # 1. Logistic Regression (baseline - no tuning)
    # ---------------------------------------------------------------
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    results.append({
        "model": "Logistic Regression",
        "best_params": list(log_reg.get_params().keys())
    })

    # ---------------------------------------------------------------
    # 2. Random Forest with RandomizedSearchCV
    # ---------------------------------------------------------------
    rf_param_dist = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"]
    }
    rf_cv = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=rf_param_dist,
        n_iter=30,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    rf_cv.fit(X_train, y_train)
    results.append({
        "model": "Random Forest",
        "best_params": list(rf_cv.best_params_.items())
    })

    # ---------------------------------------------------------------
    # 3. Decision Tree with RandomizedSearchCV
    # ---------------------------------------------------------------
    dt_param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["auto", "sqrt", "log2"]
    }
    dt_cv = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_distributions=dt_param_dist,
        n_iter=30,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    dt_cv.fit(X_train, y_train)
    results.append({
        "model": "Decision Tree",
        "best_params": list(dt_cv.best_params_.items())
    })

    # ---------------------------------------------------------------
    # 4. KNN with RandomizedSearchCV
    # ---------------------------------------------------------------
    knn_param_dist = {
        "n_neighbors": list(range(1, 31)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2]
    }
    knn_cv = RandomizedSearchCV(
        KNeighborsClassifier(),
        param_distributions=knn_param_dist,
        n_iter=30,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    knn_cv.fit(X_train, y_train)
    results.append({
        "model": "KNN",
        "best_params": list(knn_cv.best_params_.items())
    })

    # ---------------------------------------------------------------
    # 5. Gaussian Naive Bayes with GridSearchCV
    # ---------------------------------------------------------------
    nb_param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    nb_cv = GridSearchCV(
        GaussianNB(),
        param_grid=nb_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    nb_cv.fit(X_train, y_train)
    results.append({
        "model": "Naive Bayes",
        "best_params": list(nb_cv.best_params_.items())
    })

    # ---------------------------------------------------------------
    # Assemble results into a DataFrame
    # ---------------------------------------------------------------
    return pd.DataFrame(results)