# NHL Stats Analysis System

A Python-based system for collecting and analyzing NHL player statistics. The system fetches current and historical stats for players from multiple teams and processes them into a format suitable for machine learning and statistical analysis.

## Features

- **Data Collection**
  - Fetches player data from the NHL API
  - Currently supports Toronto Maple Leafs and Florida Panthers
  - Collects both current season and career statistics

- **Data Processing**
  - Processes raw JSON data into structured format
  - Calculates per-game metrics
  - Generates 27 features per player including:
    * Basic info (ID, age, position)
    * Physical attributes (height, weight)
    * Current season stats
    * Career statistics
    * Special teams performance

## Project Structure

```
nhl-stats/
├── data/                      # Processed data output
│   └── processed_player_stats.csv
├── src/
│   └── nhl_ml/
│       ├── api/              # NHL API interaction
│       │   └── nhl_api.py
│       └── ml/               # Data processing
│           └── data_processor.py
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nhl-stats
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Fetch player data:
```bash
python src/nhl_ml/api/nhl_api.py
```

2. Process the data:
```bash
python src/nhl_ml/ml/data_processor.py
```

The processed data will be saved to `data/processed_player_stats.csv`.

## Data Features

The system collects and processes the following features for each player:

- Basic Information:
  * Player ID
  * Name
  * Team
  * Position
  * Age
  * Height (cm)
  * Weight (kg)

- Current Season Statistics:
  * Games Played
  * Goals
  * Assists
  * Points
  * Plus/Minus
  * Penalty Minutes
  * Shots
  * Shooting Percentage
  * Power Play Goals
  * Power Play Points

- Career Statistics:
  * All of the above for career totals

- Calculated Metrics:
  * Goals per Game
  * Points per Game
  * Shots per Game

## License

MIT License 