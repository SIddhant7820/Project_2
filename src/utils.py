import os
import joblib
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

from sklearn.metrics import r2_score
import sys
from src.exception import CustomException

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            # Train model
            model.fit(x_train, y_train)

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save ONLY test score (used for model selection)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
