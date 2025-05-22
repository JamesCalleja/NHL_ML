"""
Module for training ML models on NHL data.
"""
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

from .data_processor import MLDataProcessor


class ModelTrainer:
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize the model trainer.

        Args:
            model_type (str): Type of model to train ('random_forest' or
                'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.data_processor = MLDataProcessor()

    def create_model(self) -> None:
        """Create the specified type of model with default parameters."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}"
            )

    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Train the model on the provided data.

        Args:
            data (pd.DataFrame): Training data
            target_col (str): Name of the target column

        Returns:
            Dict[str, float]: Dictionary containing training metrics
        """
        # Prepare the data
        X_train, X_test, y_train, y_test = (
            self.data_processor.prepare_features(data, target_col)
        )

        # Create and train the model
        self.create_model()
        self.model.fit(X_train, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath (str): Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save the model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.data_processor.scaler,
            'feature_columns': self.data_processor.feature_columns,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> 'ModelTrainer':
        """Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model

        Returns:
            ModelTrainer: Instance with loaded model and scaler
        """
        model_data = joblib.load(filepath)

        trainer = cls(model_type=model_data['model_type'])
        trainer.model = model_data['model']
        trainer.data_processor.scaler = model_data['scaler']
        trainer.data_processor.feature_columns = (
            model_data['feature_columns']
        )

        return trainer

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model.

        Returns:
            pd.DataFrame: DataFrame containing feature names
                and their importance
        """
        if self.model is None:
            raise ValueError(
                "No model to save. Train a model first."
            )

        dp = self.data_processor
        return dp.create_feature_importance_df(
            self.model,
            dp.feature_columns
        )
