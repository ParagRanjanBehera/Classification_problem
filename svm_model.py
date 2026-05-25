"""Support Vector Machine (SVM) Classifier."""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils import load_sample_data, scale_features, evaluate_model, perform_cross_validation


class SVMClassifier:
    """Support Vector Machine Classifier wrapper."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 gamma: str = 'scale', random_state: int = 42):
        """
        Initialize SVM Classifier.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            random_state: Random state for reproducibility
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
        self.params = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model.
        
        Args:
            X_train: Training features (should be scaled)
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        print("SVM model trained successfully.")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features (should be scaled)
        
        Returns:
            Predicted labels
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X_test: Test features (should be scaled)
        
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X_test)
    
    def get_support_vectors(self) -> np.ndarray:
        """
        Get support vectors.
        
        Returns:
            Support vectors
        """
        return self.model.support_vectors_
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features (should be scaled)
            y_train: Training labels
        
        Returns:
            Best parameters and best score
        """
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
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


def test_svm():
    """Test SVM Classifier."""
    print("\n" + "="*70)
    print("SUPPORT VECTOR MACHINE (SVM) CLASSIFIER TESTS")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_data()
    
    # Scale features (important for SVM)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Initialize and train model
    svm_classifier = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
    svm_classifier.train(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm_classifier.predict(X_test_scaled)
    y_pred_proba = svm_classifier.predict_proba(X_test_scaled)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "SVM (RBF)")
    
    # Cross-validation
    print("\nPerforming 5-Fold Cross-Validation...")
    cv_results = perform_cross_validation(svm_classifier.model, X_train_scaled, y_train, cv=5)
    print(f"Mean CV Score: {cv_results['Mean CV Score']:.4f}")
    print(f"Std CV Score: {cv_results['Std CV Score']:.4f}")
    print(f"Individual Scores: {cv_results['Individual Scores']}")
    
    # Support vectors info
    print(f"\nNumber of Support Vectors: {len(svm_classifier.get_support_vectors())}")
    
    # Hyperparameter tuning
    print("\nPerforming Hyperparameter Tuning...")
    tuning_results = svm_classifier.hyperparameter_tuning(X_train_scaled, y_train)
    print(f"Best Parameters: {tuning_results['best_params']}")
    print(f"Best Score: {tuning_results['best_score']:.4f}")
    
    # Re-evaluate with best parameters
    y_pred_tuned = svm_classifier.predict(X_test_scaled)
    metrics_tuned = evaluate_model(y_test, y_pred_tuned, None, "SVM (Tuned)")
    
    return {
        'model': svm_classifier,
        'scaler': scaler,
        'metrics': metrics,
        'metrics_tuned': metrics_tuned,
        'cv_results': cv_results
    }


if __name__ == "__main__":
    test_svm()
