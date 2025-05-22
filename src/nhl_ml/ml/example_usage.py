"""
Example usage of the NHL ML components for player point prediction.
"""
import pandas as pd
from pathlib import Path
import os

from .train_model import ModelTrainer
from .model_evaluation import ModelEvaluator


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for player point prediction."""
    # Select relevant features
    feature_cols = [
        'age', 'height_cm', 'weight_kg', 'games_played',
        'shots', 'shooting_pct', 'powerplay_goals',
        'career_games', 'career_goals', 'career_assists',
        'career_points', 'career_plus_minus', 'career_shots',
        'career_shooting_pct', 'career_powerplay_goals',
        'goals_per_game', 'shots_per_game'
    ]

    # Filter out goalies and players with no games played
    df = df[df['position'] != 'G']
    df = df[df['games_played'] > 0]

    return df[feature_cols + ['points']]


def run_example():
    """Run an example ML pipeline using NHL player data."""
    # Get the absolute path to the workspace root
    workspace_root = (
        Path(os.path.abspath(__file__))
        .parents[3]
    )

    # Load player data
    data_path = workspace_root / "data/processed_player_stats.csv"
    if not data_path.exists():
        print(
            f"Please ensure {data_path} exists with processed player "
            "statistics"
        )
        return

    data = pd.read_csv(data_path)

    # Prepare features
    prepared_data = prepare_features(data)

    # Initialize trainer and evaluator
    trainer = ModelTrainer(model_type='random_forest')
    evaluator = ModelEvaluator()

    # Train the model to predict points
    target_col = 'points'
    metrics = trainer.train(prepared_data, target_col)

    # Print training metrics
    print("\nTraining Metrics for Point Prediction:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Get feature importance
    importance_df = trainer.get_feature_importance()
    print("\nTop 10 Most Important Features for Point Prediction:")
    print("-" * 50)
    print(importance_df.head(10))

    # Save the model
    model_path = workspace_root / "models/nhl_point_predictor.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Load the model and make predictions
    loaded_trainer = ModelTrainer.load_model(str(model_path))

    # Process new data for predictions (using a sample of players)
    sample_players = (
        data[data['games_played'] >= 40]
        .sample(n=10, random_state=42)
    )
    print("\nPredicting points for sample players:")
    print("-" * 40)

    # Prepare sample data
    sample_features = prepare_features(sample_players)
    processed_data = loaded_trainer.data_processor.process_new_data(
        sample_features
    )
    predictions = loaded_trainer.model.predict(processed_data)

    # Create comparison DataFrame
    results = pd.DataFrame({
        'Name': sample_players['name'],
        'Position': sample_players['position'],
        'Actual Points': sample_players['points'],
        'Predicted Points': predictions.round(1)
    })
    print(results.to_string(index=False))

    # Evaluate predictions
    actual = sample_features[target_col].values
    eval_metrics = evaluator.evaluate_predictions(actual, predictions)

    print("\nEvaluation Metrics on Sample Players:")
    print("-" * 40)
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create and save plots
    plots_dir = workspace_root / "reports/figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Predictions plot
    fig, _ = evaluator.plot_predictions("NHL Player Point Predictions")
    fig.savefig(plots_dir / "point_predictions.png")

    # Feature importance plot
    fig, _ = evaluator.plot_feature_importance(importance_df)
    fig.savefig(plots_dir / "point_feature_importance.png")

    # Residuals plot
    fig, _ = evaluator.plot_residuals()
    fig.savefig(plots_dir / "point_residuals.png")

    print(f"\nPlots saved to {plots_dir}")


if __name__ == "__main__":
    run_example()
