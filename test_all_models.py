"""Comprehensive validation tests for all classification models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import load_sample_data, scale_features, evaluate_model
from decision_tree import DTClassifier
from random_forest import RFClassifier
from xgboost_model import XGBClassifier
from lightgbm_model import LGBClassifier
from svm_model import SVMClassifier
from deep_learning import NeuralNetworkClassifier


class TestDecisionTree:
    """Test Decision Tree Classifier."""
    
    @pytest.fixture
    def setup_data(self):
        X_train, X_test, y_train, y_test = load_sample_data()
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self):
        """Test model initialization."""
        model = DTClassifier(max_depth=10)
        assert model.model is not None
        assert model.params['max_depth'] == 10
    
    def test_training(self, setup_data):
        """Test model training."""
        X_train, _, y_train, _ = setup_data
        model = DTClassifier()
        model.train(X_train, y_train)
        assert model.model.tree_ is not None
    
    def test_prediction(self, setup_data):
        """Test model predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = DTClassifier()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
        assert len(np.unique(predictions)) > 0
    
    def test_predict_proba(self, setup_data):
        """Test probability predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = DTClassifier()
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == X_test.shape[0]
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_feature_importance(self, setup_data):
        """Test feature importance extraction."""
        X_train, _, y_train, _ = setup_data
        model = DTClassifier()
        model.train(X_train, y_train)
        importances = model.get_feature_importance()
        assert importances.shape[0] == X_train.shape[1]
        assert np.allclose(importances.sum(), 1.0, atol=1e-5)
    
    def test_hyperparameter_tuning(self, setup_data):
        """Test hyperparameter tuning."""
        X_train, _, y_train, _ = setup_data
        model = DTClassifier()
        results = model.hyperparameter_tuning(X_train, y_train)
        assert 'best_params' in results
        assert 'best_score' in results
        assert results['best_score'] >= 0
        assert results['best_score'] <= 1


class TestRandomForest:
    """Test Random Forest Classifier."""
    
    @pytest.fixture
    def setup_data(self):
        X_train, X_test, y_train, y_test = load_sample_data()
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self):
        """Test model initialization."""
        model = RFClassifier(n_estimators=100)
        assert model.model is not None
        assert model.params['n_estimators'] == 100
    
    def test_training(self, setup_data):
        """Test model training."""
        X_train, _, y_train, _ = setup_data
        model = RFClassifier(n_estimators=50)
        model.train(X_train, y_train)
        assert model.model.n_estimators == 50
    
    def test_prediction(self, setup_data):
        """Test model predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = RFClassifier(n_estimators=50)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
        assert len(np.unique(predictions)) > 0
    
    def test_predict_proba(self, setup_data):
        """Test probability predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = RFClassifier(n_estimators=50)
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == X_test.shape[0]
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_feature_importance(self, setup_data):
        """Test feature importance extraction."""
        X_train, _, y_train, _ = setup_data
        model = RFClassifier(n_estimators=50)
        model.train(X_train, y_train)
        importances = model.get_feature_importance()
        assert importances.shape[0] == X_train.shape[1]
        assert np.allclose(importances.sum(), 1.0, atol=1e-5)


class TestXGBoost:
    """Test XGBoost Classifier."""
    
    @pytest.fixture
    def setup_data(self):
        X_train, X_test, y_train, y_test = load_sample_data()
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self):
        """Test model initialization."""
        try:
            model = XGBClassifier(n_estimators=100)
            assert model.model is not None
            assert model.params['n_estimators'] == 100
        except ImportError:
            pytest.skip("XGBoost not installed")
    
    def test_training(self, setup_data):
        """Test model training."""
        try:
            X_train, _, y_train, _ = setup_data
            model = XGBClassifier(n_estimators=50)
            model.train(X_train, y_train)
            assert model.model is not None
        except ImportError:
            pytest.skip("XGBoost not installed")
    
    def test_prediction(self, setup_data):
        """Test model predictions."""
        try:
            X_train, X_test, y_train, _ = setup_data
            model = XGBClassifier(n_estimators=50)
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            assert predictions.shape[0] == X_test.shape[0]
        except ImportError:
            pytest.skip("XGBoost not installed")


class TestLightGBM:
    """Test LightGBM Classifier."""
    
    @pytest.fixture
    def setup_data(self):
        X_train, X_test, y_train, y_test = load_sample_data()
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self):
        """Test model initialization."""
        try:
            model = LGBClassifier(n_estimators=100)
            assert model.model is not None
            assert model.params['n_estimators'] == 100
        except ImportError:
            pytest.skip("LightGBM not installed")
    
    def test_training(self, setup_data):
        """Test model training."""
        try:
            X_train, _, y_train, _ = setup_data
            model = LGBClassifier(n_estimators=50)
            model.train(X_train, y_train)
            assert model.model is not None
        except ImportError:
            pytest.skip("LightGBM not installed")
    
    def test_prediction(self, setup_data):
        """Test model predictions."""
        try:
            X_train, X_test, y_train, _ = setup_data
            model = LGBClassifier(n_estimators=50)
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            assert predictions.shape[0] == X_test.shape[0]
        except ImportError:
            pytest.skip("LightGBM not installed")


class TestSVM:
    """Test SVM Classifier."""
    
    @pytest.fixture
    def setup_data(self):
        X_train, X_test, y_train, y_test = load_sample_data()
        X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def test_initialization(self):
        """Test model initialization."""
        model = SVMClassifier(kernel='rbf')
        assert model.model is not None
        assert model.params['kernel'] == 'rbf'
    
    def test_training(self, setup_data):
        """Test model training."""
        X_train, _, y_train, _ = setup_data
        model = SVMClassifier(kernel='rbf')
        model.train(X_train, y_train)
        assert model.model is not None
    
    def test_prediction(self, setup_data):
        """Test model predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = SVMClassifier(kernel='rbf')
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_predict_proba(self, setup_data):
        """Test probability predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = SVMClassifier(kernel='rbf')
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == X_test.shape[0]
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_support_vectors(self, setup_data):
        """Test support vectors retrieval."""
        X_train, _, y_train, _ = setup_data
        model = SVMClassifier(kernel='rbf')
        model.train(X_train, y_train)
        sv = model.get_support_vectors()
        assert sv.shape[1] == X_train.shape[1]


class TestNeuralNetwork:
    """Test Neural Network Classifier."""
    
    @pytest.fixture
    def setup_data(self):
        X_train, X_test, y_train, y_test = load_sample_data()
        X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def test_initialization(self):
        """Test model initialization."""
        model = NeuralNetworkClassifier(
            input_dim=20,
            hidden_units=[128, 64, 32]
        )
        assert model.model is not None
    
    def test_training(self, setup_data):
        """Test model training."""
        X_train, _, y_train, _ = setup_data
        model = NeuralNetworkClassifier(
            input_dim=X_train.shape[1],
            hidden_units=[64, 32]
        )
        model.train(X_train, y_train, epochs=5)
        assert model.history is not None
    
    def test_prediction(self, setup_data):
        """Test model predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = NeuralNetworkClassifier(
            input_dim=X_train.shape[1],
            hidden_units=[64, 32]
        )
        model.train(X_train, y_train, epochs=5)
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_predict_proba(self, setup_data):
        """Test probability predictions."""
        X_train, X_test, y_train, _ = setup_data
        model = NeuralNetworkClassifier(
            input_dim=X_train.shape[1],
            hidden_units=[64, 32]
        )
        model.train(X_train, y_train, epochs=5)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == X_test.shape[0]
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_model_comparison():
    """Test and compare all models."""
    print("\n" + "="*70)
    print("COMPARING ALL MODELS")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_sample_data()
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    results = {}
    
    # Decision Tree
    dt = DTClassifier()
    dt.train(X_train, y_train)
    dt_pred = dt.predict(X_test)
    results['Decision Tree'] = accuracy_score(y_test, dt_pred)
    
    # Random Forest
    rf = RFClassifier(n_estimators=50)
    rf.train(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, rf_pred)
    
    # SVM
    svm = SVMClassifier(kernel='rbf')
    svm.train(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    results['SVM'] = accuracy_score(y_test, svm_pred)
    
    # XGBoost
    try:
        xgb = XGBClassifier(n_estimators=50)
        xgb.train(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results['XGBoost'] = accuracy_score(y_test, xgb_pred)
    except ImportError:
        results['XGBoost'] = 'Not installed'
    
    # LightGBM
    try:
        lgb = LGBClassifier(n_estimators=50)
        lgb.train(X_train, y_train)
        lgb_pred = lgb.predict(X_test)
        results['LightGBM'] = accuracy_score(y_test, lgb_pred)
    except ImportError:
        results['LightGBM'] = 'Not installed'
    
    print("\nModel Comparison Results (Accuracy):")
    print("-" * 50)
    for model_name, accuracy in results.items():
        if isinstance(accuracy, float):
            print(f"{model_name:20s}: {accuracy:.4f}")
        else:
            print(f"{model_name:20s}: {accuracy}")
    
    return results


if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, '-v'])
    
    # Run model comparison
    test_model_comparison()
