"""XGBoost Classifier."""

import numpy as np
try:
    import xgboost as xgb
except ImportError:
    print("Please install xgboost: pip install xgboost")

from sklearn.model_selection import GridSearchCV
from utils import load_sample_data, evaluate_model, perform_cross_validation


class XGBClassifier:
    """XGBoost Classifier wrapper."""
    
    def __init__(self, max_depth: int = 6, learning_rate: float = 0.3,
                 n_estimators: int = 100, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        """
        Initialize XGBoost Classifier.
        
        Args:
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            random_state: Random state for reproducibility
        """
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        print("XGBoost model trained successfully.")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features
        
        Returns:
            Predicted labels
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X_test: Test features
        
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        return self.model.feature_importances_
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Best parameters and best score
        """
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.1, 0.3, 0.5],
            'n_estimators': [50, 100, 200],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }


def test_xgboost():
    """Test XGBoost Classifier."""
    print("\n" + "="*70)
    print("XGBOOST CLASSIFIER TESTS")
    print("="*70)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_sample_data()
        
        # Initialize and train model
        xgb_classifier = XGBClassifier(
            max_depth=6,
            learning_rate=0.3,
            n_estimators=100
        )
        xgb_classifier.train(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_classifier.predict(X_test)
        y_pred_proba = xgb_classifier.predict_proba(X_test)
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        # Cross-validation
        print("\nPerforming 5-Fold Cross-Validation...")
        cv_results = perform_cross_validation(xgb_classifier.model, X_train, y_train, cv=5)
        print(f"Mean CV Score: {cv_results['Mean CV Score']:.4f}")
        print(f"Std CV Score: {cv_results['Std CV Score']:.4f}")
        print(f"Individual Scores: {cv_results['Individual Scores']}")
        
        # Feature importance
        print("\nTop 5 Important Features:")
        importances = xgb_classifier.get_feature_importance()
        top_indices = np.argsort(importances)[-5:][::-1]
        for idx, importance in enumerate(importances[top_indices]):
            print(f"Feature {top_indices[idx]}: {importance:.4f}")
        
        # Hyperparameter tuning
        print("\nPerforming Hyperparameter Tuning...")
        tuning_results = xgb_classifier.hyperparameter_tuning(X_train, y_train)
        print(f"Best Parameters: {tuning_results['best_params']}")
        print(f"Best Score: {tuning_results['best_score']:.4f}")
        
        # Re-evaluate with best parameters
        y_pred_tuned = xgb_classifier.predict(X_test)
        metrics_tuned = evaluate_model(y_test, y_pred_tuned, None, "XGBoost (Tuned)")
        
        return {
            'model': xgb_classifier,
            'metrics': metrics,
            'metrics_tuned': metrics_tuned,
            'cv_results': cv_results
        }
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install xgboost using: pip install xgboost")
        return None


if __name__ == "__main__":
    test_xgboost()
