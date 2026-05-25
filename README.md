# Classification Models Repository

A comprehensive collection of machine learning classification models with implementation in Python using scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, and more.

## Features

### Traditional ML Models
- **Decision Tree**: Simple and interpretable tree-based classifier
- **Random Forest**: Ensemble of decision trees for improved accuracy
- **Support Vector Machine (SVM)**: Kernel-based classifier for complex decision boundaries

### Gradient Boosting Models
- **XGBoost**: Extreme Gradient Boosting for high performance
- **LightGBM**: Light Gradient Boosting Machine for speed and efficiency

### Deep Learning Models
- **Neural Network**: Fully connected neural network with batch normalization and dropout
- **Convolutional Neural Network (CNN)**: For image classification tasks

## Project Structure

```
├── utils.py                 # Utility functions for data loading, scaling, and evaluation
├── decision_tree.py         # Decision Tree Classifier implementation
├── random_forest.py         # Random Forest Classifier implementation
├── xgboost_model.py         # XGBoost Classifier implementation
├── lightgbm_model.py        # LightGBM Classifier implementation
├── svm_model.py             # SVM Classifier implementation
├── deep_learning.py         # Neural Network and CNN implementations
├── test_all_models.py       # Comprehensive pytest-based validation tests
└── README.md                # This file
```

## Installation

### Required Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Optional Dependencies

```bash
# For XGBoost
pip install xgboost

# For LightGBM
pip install lightgbm

# For Deep Learning
pip install tensorflow keras

# For testing
pip install pytest
```

### Install All

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm tensorflow pytest
```

## Usage

### Quick Start - Decision Tree

```python
from decision_tree import DTClassifier
from utils import load_sample_data, evaluate_model

# Load sample data
X_train, X_test, y_train, y_test = load_sample_data()

# Initialize and train model
dt_model = DTClassifier(max_depth=10)
dt_model.train(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate
metrics = evaluate_model(y_test, y_pred, model_name="Decision Tree")
```

### Random Forest Example

```python
from random_forest import RFClassifier
from utils import load_sample_data, perform_cross_validation

X_train, X_test, y_train, y_test = load_sample_data()

rf_model = RFClassifier(n_estimators=100, max_depth=15)
rf_model.train(X_train, y_train)

y_pred = rf_model.predict(X_test)
feature_importance = rf_model.get_feature_importance()

# Cross-validation
cv_results = perform_cross_validation(rf_model.model, X_train, y_train, cv=5)
print(f"Mean CV Score: {cv_results['Mean CV Score']:.4f}")
```

### SVM with Feature Scaling

```python
from svm_model import SVMClassifier
from utils import load_sample_data, scale_features

X_train, X_test, y_train, y_test = load_sample_data()

# Scale features (important for SVM)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

svm_model = SVMClassifier(kernel='rbf', C=1.0)
svm_model.train(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)
```

### XGBoost with Hyperparameter Tuning

```python
from xgboost_model import XGBClassifier
from utils import load_sample_data

X_train, X_test, y_train, y_test = load_sample_data()

xgb_model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=100)
xgb_model.train(X_train, y_train)

# Hyperparameter tuning
best_params = xgb_model.hyperparameter_tuning(X_train, y_train)
print(f"Best parameters: {best_params['best_params']}")
print(f"Best score: {best_params['best_score']:.4f}")
```

### Neural Network Example

```python
from deep_learning import NeuralNetworkClassifier
from utils import load_sample_data, scale_features

X_train, X_test, y_train, y_test = load_sample_data()
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

nn_model = NeuralNetworkClassifier(
    input_dim=X_train_scaled.shape[1],
    hidden_units=[128, 64, 32],
    dropout_rate=0.3,
    learning_rate=0.001
)

nn_model.train(X_train_scaled, y_train, epochs=50, batch_size=32)
y_pred = nn_model.predict(X_test_scaled)
```

## Running Tests

### Run all tests with pytest

```bash
pytest test_all_models.py -v
```

### Run specific model tests

```bash
pytest test_all_models.py::TestDecisionTree -v
pytest test_all_models.py::TestRandomForest -v
pytest test_all_models.py::TestSVM -v
pytest test_all_models.py::TestXGBoost -v
pytest test_all_models.py::TestLightGBM -v
pytest test_all_models.py::TestNeuralNetwork -v
```

### Run model comparison

```bash
python test_all_models.py
```

### Run individual model tests

```bash
python decision_tree.py
python random_forest.py
python svm_model.py
python xgboost_model.py
python lightgbm_model.py
python deep_learning.py
```

## Model Evaluation Metrics

Each model includes the following evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve (for binary classification)
- **Confusion Matrix**: Visual representation of prediction results
- **Classification Report**: Detailed per-class metrics

## Hyperparameter Tuning

All models support hyperparameter tuning using GridSearchCV:

```python
model = DTClassifier()
tuning_results = model.hyperparameter_tuning(X_train, y_train)
print(f"Best parameters: {tuning_results['best_params']}")
print(f"Best score: {tuning_results['best_score']:.4f}")
```

## Cross-Validation

Perform k-fold cross-validation to assess model performance:

```python
from utils import perform_cross_validation

cv_results = perform_cross_validation(model.model, X_train, y_train, cv=5)
print(f"Mean CV Score: {cv_results['Mean CV Score']:.4f}")
print(f"Std CV Score: {cv_results['Std CV Score']:.4f}")
```

## Key Features

### Utility Functions
- `load_sample_data()`: Generate sample classification dataset
- `scale_features()`: Standardize features (important for SVM and NN)
- `evaluate_model()`: Comprehensive model evaluation with multiple metrics
- `perform_cross_validation()`: K-fold cross-validation
- `plot_confusion_matrix()`: Visualize confusion matrix
- `plot_roc_curve()`: Plot ROC curve for binary classification

### Model Classes
- Consistent API across all models
- `train()`: Train the model
- `predict()`: Make class predictions
- `predict_proba()`: Get probability predictions
- `get_feature_importance()`: Extract feature importance (where applicable)
- `hyperparameter_tuning()`: Optimize hyperparameters

## Validation Tests

Comprehensive test suite includes:

- **Initialization Tests**: Verify model creation
- **Training Tests**: Ensure successful model training
- **Prediction Tests**: Validate prediction output shapes and values
- **Probability Tests**: Check probability predictions are valid (sum to 1)
- **Feature Importance Tests**: Verify importance scores
- **Hyperparameter Tuning Tests**: Test parameter optimization
- **Model Comparison Tests**: Compare performance across models

## Performance Optimization

### For Traditional Models
- Use `n_jobs=-1` to leverage all CPU cores
- Feature scaling for SVM
- Hyperparameter tuning for improved accuracy

### For Gradient Boosting
- Adjust learning rate and number of estimators
- Use `max_depth` to control model complexity
- Early stopping with validation set

### For Deep Learning
- Use batch normalization to stabilize training
- Apply dropout for regularization
- Use learning rate scheduling
- Leverage GPU acceleration with TensorFlow

## Common Use Cases

1. **Binary Classification**: Change output layer to 2 units
2. **Multi-class Classification**: Use 'softmax' activation (default)
3. **Imbalanced Data**: Use class weights or SMOTE
4. **Feature Selection**: Use feature importance scores
5. **Model Ensembling**: Combine predictions from multiple models

## Troubleshooting

### XGBoost/LightGBM not installed
```bash
pip install xgboost lightgbm
```

### TensorFlow issues
```bash
pip install --upgrade tensorflow
```

### Memory issues with large datasets
- Reduce batch size
- Use model.predict with smaller chunks
- Consider using LightGBM (more memory efficient)

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## Author

ParagRanjanBehera

## Acknowledgments

This repository provides practical implementations of popular classification algorithms with comprehensive testing and evaluation utilities.
