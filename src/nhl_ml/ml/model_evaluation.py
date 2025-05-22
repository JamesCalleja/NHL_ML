"""
Module for evaluating ML models on NHL data.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator."""
        self.predictions = None
        self.actual_values = None
        self.feature_importance = None

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model predictions using various metrics.

        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        self.predictions = y_pred
        self.actual_values = y_true

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }

        return metrics

    def plot_predictions(
        self, title: str = "Predicted vs Actual Values"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a scatter plot of predicted vs actual values.

        Args:
            title (str): Title for the plot

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        if self.predictions is None or self.actual_values is None:
            raise ValueError(
                "No predictions available. Run evaluate_predictions first."
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create scatter plot
        ax.scatter(self.actual_values, self.predictions, alpha=0.5)

        # Add perfect prediction line
        min_val = min(self.actual_values.min(), self.predictions.min())
        max_val = max(self.actual_values.max(), self.predictions.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            label='Perfect Prediction'
        )

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.legend()

        return fig, ax

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar plot of feature importances.

        Args:
            importance_df (pd.DataFrame): DataFrame with feature importances
            top_n (int): Number of top features to display

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        self.feature_importance = importance_df

        # Get top N features
        top_features = importance_df.nlargest(top_n, 'importance')

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bar plot
        sns.barplot(
            data=top_features,
            x='importance',
            y='feature',
            ax=ax
        )

        ax.set_title(f'Top {top_n} Most Important Features')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')

        return fig, ax

    def plot_residuals(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a residual plot.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        if self.predictions is None or self.actual_values is None:
            raise ValueError(
                "No predictions available. Run evaluate_predictions first."
            )

        residuals = self.actual_values - self.predictions

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create residual plot
        ax.scatter(self.predictions, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')

        return fig, ax

    def get_prediction_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of predictions vs actual values.

        Returns:
            pd.DataFrame: Summary DataFrame
        """
        if self.predictions is None or self.actual_values is None:
            raise ValueError(
                "No predictions available. Run evaluate_predictions first."
            )

        return pd.DataFrame({
            'actual': self.actual_values,
            'predicted': self.predictions,
            'residual': self.actual_values - self.predictions,
            'abs_error': np.abs(self.actual_values - self.predictions)
        })
