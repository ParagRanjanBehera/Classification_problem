"""Random Forest Classifier."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils import load_sample_data, evaluate_model, perform_cross_validation, plot_confusion_matrix


class RFClassifier:
    """Random Forest Classifier wrapper."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15,
                 min_samples_split: int = 5, min_samples_leaf: int = 2,
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize Random Forest Classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples at leaf
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        print("Random Forest model trained successfully.")
    
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
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
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


def test_random_forest():
    """Test Random Forest Classifier."""
    print("\n" + "="*70)
    print("RANDOM FOREST CLASSIFIER TESTS")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_data()
    
    # Initialize and train model
    rf_classifier = RFClassifier(n_estimators=100, max_depth=15)
    rf_classifier.train(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    
    # Cross-validation
    print("\nPerforming 5-Fold Cross-Validation...")
    cv_results = perform_cross_validation(rf_classifier.model, X_train, y_train, cv=5)
    print(f"Mean CV Score: {cv_results['Mean CV Score']:.4f}")
    print(f"Std CV Score: {cv_results['Std CV Score']:.4f}")
    print(f"Individual Scores: {cv_results['Individual Scores']}")
    
    # Feature importance
    print("\nTop 5 Important Features:")
    importances = rf_classifier.get_feature_importance()
    top_indices = np.argsort(importances)[-5:][::-1]
    for idx, importance in enumerate(importances[top_indices]):
        print(f"Feature {top_indices[idx]}: {importance:.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming Hyperparameter Tuning...")
    tuning_results = rf_classifier.hyperparameter_tuning(X_train, y_train)
    print(f"Best Parameters: {tuning_results['best_params']}")
    print(f"Best Score: {tuning_results['best_score']:.4f}")
    
    # Re-evaluate with best parameters
    y_pred_tuned = rf_classifier.predict(X_test)
    metrics_tuned = evaluate_model(y_test, y_pred_tuned, None, "Random Forest (Tuned)")
    
    return {
        'model': rf_classifier,
        'metrics': metrics,
        'metrics_tuned': metrics_tuned,
        'cv_results': cv_results
    }


if __name__ == "__main__":
    test_random_forest()
