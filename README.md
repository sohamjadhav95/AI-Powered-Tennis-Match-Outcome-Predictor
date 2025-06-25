# 🎾 AI-Powered Tennis Match Outcome Predictor

## Overview
This project leverages advanced machine learning and feature engineering to predict the outcomes of ATP tennis matches. By combining historical match data, dynamic player ratings (Elo), and contextual features, the system provides accurate, explainable predictions for upcoming matches.

## Project Structure
```
AI-Powered-Tennis-Match-Outcome-Predictor/
│
├── Tennis Match Prediction/
│   ├── Advance Feature Engineeing and Hyperparamete Tuning.ipynb
│   ├── New Predictions.ipynb
│   ├── Dataset/
│   │   ├── All_Yearwise/ (Raw yearly ATP match data 1968–2024)
│   │   ├── Main Dataset.csv (Merged, cleaned dataset)
│   │   └── custumized_tennis_match_feature_rich_data.csv (Feature-rich, engineered dataset)
│   ├── Models, Features and Encoders/ (Trained models, encoders, and feature dictionaries)
│   └── Archives/ (Legacy notebooks, experiments)
└── README.md
```

## Data Sources
- **Raw ATP Match Data**: CSVs for each year (1968–2024) in `Dataset/All_Yearwise/`.
- **Main Dataset**: Cleaned and merged historical data (`Main Dataset.csv`).
- **Feature-Rich Dataset**: Engineered features for modeling (`custumized_tennis_match_feature_rich_data.csv`).

### Example Columns (from raw data):
- `tourney_date`, `surface`, `round`, `tourney_level`, `draw_size`, `best_of`
- `winner_id`, `winner_name`, `winner_hand`, `winner_ht`, `winner_age`, `winner_rank`
- `loser_id`, `loser_name`, `loser_hand`, `loser_ht`, `loser_age`, `loser_rank`

## Feature Engineering
The project goes beyond basic stats, creating advanced features that capture player form, rivalry, and context:
- **Global Elo**: Dynamic rating for each player, updated after every match.
- **Surface-Specific Elo**: Elo ratings per surface (Hard, Clay, Grass, Carpet).
- **Head-to-Head (H2H) Winrate**: Historical win rate between any two players.
- **Recent Form**: Rolling win rate over the last 10 matches.
- **Fatigue Estimate**: Number of matches played in the last 30 days.
- **Contextual Features**: Surface, round, tournament level, draw size, best-of, player handedness, height, age, and ranking.

## Modeling Approach
- **Model**: XGBoost Classifier (with experiments using Logistic Regression and LightGBM)
- **Labeling**: For each match, players are randomly assigned as Player 1/2; the label is 1 if Player 1 wins, 0 otherwise (prevents model bias).
- **Feature Encoding**: Categorical features (surface, round, level, hand) are label-encoded for model compatibility.
- **Training**: Extensive hyperparameter tuning and feature selection for optimal accuracy.
- **Evaluation**: Accuracy, confusion matrix, classification report, and feature importance plots.
- **Ensemble**: Optionally, a soft-voting ensemble of XGBoost and LightGBM for improved stability.

## Usage
### 1. Training & Feature Engineering
- See `Advance Feature Engineeing and Hyperparamete Tuning.ipynb` for full data cleaning, feature creation, and model training pipeline.
- Outputs:
  - Trained model (`xgb_model_final.pkl`)
  - Feature dictionaries (Elo, H2H, form, fatigue)
  - Label encoders and trained columns

### 2. Making Predictions
- Use `New Predictions.ipynb` for deployment-ready predictions on new matches.
- **Steps:**
  1. Load trained artifacts (model, encoders, feature dictionaries)
  2. Prepare new match data (as DataFrame or CSV)
  3. Apply feature engineering functions
  4. Encode categorical features
  5. Predict win probabilities and output results

#### Example Output
| player1_name    | player2_name      | predicted_winner_name | p1_win_probability |
|-----------------|------------------|-----------------------|--------------------|
| Taylor Fritz    | Alexander Zverev | Taylor Fritz          | 0.623              |

## Deployment
- All trained artifacts are stored in `Models, Features and Encoders/` for easy loading in production or web apps.
- The prediction pipeline is modular and can be integrated into web services, dashboards, or batch scripts.

## Notebooks & Archives
- **Advance Feature Engineeing and Hyperparamete Tuning.ipynb**: Main notebook for data prep, feature engineering, and model training.
- **New Predictions.ipynb**: Clean, deployment-ready prediction workflow.
- **Archives/**: Legacy experiments, baseline models, and historical approaches.

## Results
- **XGBoost Accuracy**: ~76–77% on test data (with full feature set)
- **Feature Importance**: Elo difference, surface Elo, H2H, recent form, and fatigue are top predictors.
- **Ensemble Accuracy**: Slight improvement with XGBoost + LightGBM voting ensemble.

## How to Extend
- Add new features (e.g., weather, player injuries, betting odds)
- Integrate with live data feeds for real-time predictions
- Build a web dashboard for user-friendly predictions

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, joblib

## License
This project is for educational and research purposes. Please cite appropriately if used in publications.
