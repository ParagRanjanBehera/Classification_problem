"""Deep Learning Models for Classification."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from utils import load_sample_data, scale_features, evaluate_model


class NeuralNetworkClassifier:
    """Neural Network Classifier using TensorFlow/Keras."""
    
    def __init__(self, input_dim: int, hidden_units: list = None,
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """
        Initialize Neural Network Classifier.
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        if hidden_units is None:
            hidden_units = [128, 64, 32]
        
        self.model = self._build_model(
            input_dim, hidden_units, dropout_rate, learning_rate
        )
        self.history = None
    
    def _build_model(self, input_dim: int, hidden_units: list,
                     dropout_rate: float, learning_rate: float):
        """
        Build neural network model.
        
        Args:
            input_dim: Input dimension
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer (3 classes)
        model.add(layers.Dense(3, activation='softmax'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        print("Neural Network model trained successfully.")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features
        
        Returns:
            Predicted labels
        """
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X_test: Test features
        
        Returns:
            Predicted probabilities
        """
        return self.model.predict(X_test, verbose=0)


class ConvolutionalNeuralNetwork:
    """Convolutional Neural Network (CNN) for image classification."""
    
    def __init__(self, input_shape: tuple = (28, 28, 1),
                 learning_rate: float = 0.001):
        """
        Initialize CNN.
        
        Args:
            input_shape: Input shape of images
            learning_rate: Learning rate for optimizer
        """
        self.model = self._build_model(input_shape, learning_rate)
        self.history = None
    
    def _build_model(self, input_shape: tuple, learning_rate: float):
        """
        Build CNN model.
        
        Args:
            input_shape: Input shape
            learning_rate: Learning rate
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 32) -> None:
        """
        Train the CNN.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        print("CNN model trained successfully.")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test images
        
        Returns:
            Predicted labels
        """
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X_test: Test images
        
        Returns:
            Predicted probabilities
        """
        return self.model.predict(X_test, verbose=0)


def test_neural_network():
    """Test Neural Network Classifier."""
    print("\n" + "="*70)
    print("NEURAL NETWORK CLASSIFIER TESTS")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_sample_data()
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Split training data into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize and train model
    nn_classifier = NeuralNetworkClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    nn_classifier.train(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Make predictions
    y_pred = nn_classifier.predict(X_test_scaled)
    y_pred_proba = nn_classifier.predict_proba(X_test_scaled)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Neural Network")
    
    return {
        'model': nn_classifier,
        'metrics': metrics
    }


def test_cnn():
    """Test Convolutional Neural Network."""
    print("\n" + "="*70)
    print("CONVOLUTIONAL NEURAL NETWORK (CNN) TESTS")
    print("="*70)
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Use subset for faster testing
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    
    # Split data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    cnn_classifier = ConvolutionalNeuralNetwork(
        input_shape=(28, 28, 1),
        learning_rate=0.001
    )
    cnn_classifier.train(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=10,
        batch_size=32
    )
    
    # Make predictions
    y_pred = cnn_classifier.predict(X_test)
    y_pred_proba = cnn_classifier.predict_proba(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "CNN (MNIST)")
    
    return {
        'model': cnn_classifier,
        'metrics': metrics
    }


if __name__ == "__main__":
    test_neural_network()
    test_cnn()
