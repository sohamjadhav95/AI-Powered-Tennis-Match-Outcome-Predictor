import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from collections import deque

# Load player names and IDs
player_df = pd.read_csv(r"Tennis Match Prediction/Dataset/Player Names and ID's.csv")
player_df['player_name'] = player_df['player_name'].str.strip()
player_name_to_id = dict(zip(player_df['player_name'], player_df['player_id']))

# Load trained artifacts
model = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/xgb_model_final.pkl")
trained_columns = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/trained_columns.pkl")
encoders = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/label_encoders.pkl")
global_elo_db = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/global_elo_final.pkl")
surface_elo_db = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/surface_elo_final.pkl")
h2h_db = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/h2h_record_final.pkl")
form_db = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/recent_results_final.pkl")
fatigue_db = joblib.load(r"Tennis Match Prediction/Models, Features and Encoders/match_history_final.pkl")

BASE_ELO = 1500

st.title("🎾 Tennis Match Outcome Predictor")
st.write("Fill in the match details and get the predicted winner!")

# Options for round and tournament level
round_options = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
tourney_level_options = [
    ("G", "Grand Slam"),
    ("M", "ATP 1000"),
    ("A", "ATP 500"),
    ("B", "ATP 250"),
    ("C", "Challenger"),
    ("D", "Davis Cup")
]
tourney_level_labels = [f"{code} - {label}" for code, label in tourney_level_options]
tourney_level_code_map = {f"{code} - {label}": code for code, label in tourney_level_options}

def get_tourney_level_index(default_code="M"):
    for i, (code, label) in enumerate(tourney_level_options):
        if code == default_code:
            return i
    return 0

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        player1_name = st.text_input("Player 1 Name", "", key="p1_name")
        # Autofill Player 1 ID
        auto_player1_id = str(player_name_to_id.get(player1_name.strip(), "")) if player1_name.strip() else ""
        player1_id = st.text_input("Player 1 ID", value=auto_player1_id, key="p1_id")
        p1_rank = st.number_input("Player 1 Rank", min_value=1, step=1)
        p1_hand = st.selectbox("Player 1 Hand", options=["R", "L"], index=0)
        p1_ht = st.number_input("Player 1 Height (cm)", min_value=100, max_value=250, value=185)
        p1_age = st.number_input("Player 1 Age", min_value=10.0, max_value=60.0, value=25.0)
    with col2:
        player2_name = st.text_input("Player 2 Name", "", key="p2_name")
        # Autofill Player 2 ID
        auto_player2_id = str(player_name_to_id.get(player2_name.strip(), "")) if player2_name.strip() else ""
        player2_id = st.text_input("Player 2 ID", value=auto_player2_id, key="p2_id")
        p2_rank = st.number_input("Player 2 Rank", min_value=1, step=1)
        p2_hand = st.selectbox("Player 2 Hand", options=["R", "L"], index=0)
        p2_ht = st.number_input("Player 2 Height (cm)", min_value=100, max_value=250, value=185)
        p2_age = st.number_input("Player 2 Age", min_value=10.0, max_value=60.0, value=25.0)

    tourney_date = st.date_input("Tournament Date", value=datetime.date.today())
    surface = st.selectbox("Surface", options=["Hard", "Clay", "Grass", "Carpet"])
    round_ = st.selectbox("Round", options=round_options, index=round_options.index("F"))
    tourney_level_label = st.selectbox("Tournament Level", options=tourney_level_labels, index=get_tourney_level_index("M"))
    tourney_level = tourney_level_code_map[tourney_level_label]
    draw_size = st.number_input("Draw Size", min_value=2, value=32)
    best_of = st.selectbox("Best Of Sets", options=[3, 5], index=0)

    submitted = st.form_submit_button("Predict Winner")

if submitted:
    # Prepare input DataFrame
    new_match = {
        'tourney_date':   [pd.to_datetime(tourney_date)],
        'surface':        [surface],
        'round':          [round_],
        'tourney_level':  [tourney_level],
        'draw_size':      [draw_size],
        'best_of':        [best_of],
        'player1_id':     [int(player1_id) if player1_id else None],
        'player2_id':     [int(player2_id) if player2_id else None],
        'player1_name':   [player1_name],
        'player2_name':   [player2_name],
        'p1_rank':        [p1_rank],
        'p2_rank':        [p2_rank],
        'p1_hand':        [p1_hand],
        'p2_hand':        [p2_hand],
        'p1_ht':          [p1_ht],
        'p2_ht':          [p2_ht],
        'p1_age':         [p1_age],
        'p2_age':         [p2_age]
    }
    df_new = pd.DataFrame(new_match)

    # Feature engineering (adapted from notebook)
    def add_engineered_features(df):
        df['p1_global_elo'] = df['player1_id'].map(lambda x: global_elo_db.get(x, BASE_ELO))
        df['p2_global_elo'] = df['player2_id'].map(lambda x: global_elo_db.get(x, BASE_ELO))
        df['elo_diff'] = df['p1_global_elo'] - df['p2_global_elo']

        def get_surface_elo_diff(row):
            p1_surface = surface_elo_db.get(row['player1_id'], {}).get(row['surface'], BASE_ELO)
            p2_surface = surface_elo_db.get(row['player2_id'], {}).get(row['surface'], BASE_ELO)
            return p1_surface - p2_surface
        df['surface_elo_diff'] = df.apply(get_surface_elo_diff, axis=1)

        def get_h2h_winrate(row):
            p1_id, p2_id = row['player1_id'], row['player2_id']
            pair = tuple(sorted([p1_id, p2_id]))
            p1_wins_hist, total_matches = h2h_db.get(pair, (0, 0))
            if total_matches == 0: return 0.5
            winrate = (p1_wins_hist / total_matches) if p1_id < p2_id else ((total_matches - p1_wins_hist) / total_matches)
            return winrate
        df['h2h_winrate'] = df.apply(get_h2h_winrate, axis=1)

        def get_form(player_id):
            hist = form_db.get(player_id, deque(maxlen=10))
            return sum(hist) / len(hist) if hist else 0.5
        df['form_diff'] = df['player1_id'].map(get_form) - df['player2_id'].map(get_form)

        def get_fatigue_diff(row):
            match_date = row['tourney_date']
            thirty_days_ago = match_date - pd.Timedelta(days=30)
            p1_fatigue = sum(1 for date in fatigue_db.get(row['player1_id'], []) if date > thirty_days_ago)
            p2_fatigue = sum(1 for date in fatigue_db.get(row['player2_id'], []) if date > thirty_days_ago)
            return p1_fatigue - p2_fatigue
        df['fatigue_diff'] = df.apply(get_fatigue_diff, axis=1)

        return df

    df_pred = add_engineered_features(df_new.copy())

    for col, encoder in encoders.items():
        if col == 'hand':
            df_pred['p1_hand'] = encoder.transform(df_pred['p1_hand'])
            df_pred['p2_hand'] = encoder.transform(df_pred['p2_hand'])
        elif col in df_pred.columns:
            df_pred[col] = encoder.transform(df_pred[col])
    if 'level' in encoders and 'tourney_level' in df_pred.columns:
        df_pred['tourney_level'] = encoders['level'].transform(df_pred['tourney_level'])

    X_pred = df_pred[trained_columns]
    probs = model.predict_proba(X_pred)

    df_new['p1_win_probability'] = probs[:, 1]
    df_new['p2_win_probability'] = probs[:, 0]
    df_new['predicted_winner_name'] = np.where(df_new['p1_win_probability'] > df_new['p2_win_probability'],
                                               df_new['player1_name'], df_new['player2_name'])

    st.subheader("Prediction Results")
    st.write(df_new[['player1_name', 'player2_name', 'p1_win_probability', 'p2_win_probability', 'predicted_winner_name']].round(3))
    st.success(f"Predicted winner: {df_new.loc[0, 'predicted_winner_name']}") 