import streamlit as st
import pandas as pd
import joblib
import logging
from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(filename='nba_model.log', level=logging.INFO)

st.title("NBA Betting Prediction App (nba_api)")
st.write("Predict NBA game outcomes and player stats (points, assists, rebounds).")

# Fetch NBA data using nba_api
def fetch_data():
    logging.info("Starting data fetch with nba_api...")

    try:
        # Fetch game data
        gamefinder = leaguegamefinder.LeagueGameFinder(season_type_nullable='Regular Season')
        games_data = gamefinder.get_data_frames()[0]
        games_data.to_csv('nba_games.csv', index=False)
        logging.info(f"Fetched {len(games_data)} games. Saved to nba_games.csv.")

        # Fetch player stats
        players = {'LeBron James': '2544', 'Stephen Curry': '201939'}
        all_player_stats = []

        for player_name, player_id in players.items():
            logging.info(f"Fetching stats for {player_name} (ID: {player_id})...")
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season='2024-25',
                    season_type_all_star='Regular Season'
                )
                stats = gamelog.get_data_frames()[0]
                all_player_stats.append(stats)
                logging.info(f"Fetched stats for {player_name}.")
            except Exception as e:
                logging.error(f"Failed to fetch stats for {player_name}: {e}")
                st.error(f"Failed to fetch stats for {player_name}.")

        if all_player_stats:
            player_stats = pd.concat(all_player_stats)
            player_stats.to_csv('nba_player_stats.csv', index=False)
            logging.info("Player stats saved to nba_player_stats.csv.")
            st.success("Data fetching completed successfully!")
        else:
            logging.error("No player stats fetched.")
            st.error("Failed to fetch player stats. Check logs.")

    except Exception as e:
        logging.error(f"Data fetching error: {e}")
        st.error("Data fetching failed. Check logs.")

# Preprocess data
def preprocess_data():
    logging.info("Preprocessing data...")

    try:
        games_data = pd.read_csv('nba_games.csv')
        player_stats = pd.read_csv('nba_player_stats.csv')
        logging.info(f"Games shape: {games_data.shape}, Player stats shape: {player_stats.shape}")

        required_columns = ['GAME_ID', 'MATCHUP', 'WL', 'PTS', 'AST', 'REB']
        missing_columns = [col for col in required_columns if col not in games_data.columns]

        if missing_columns:
            logging.error(f"Missing game columns: {missing_columns}")
            st.error(f"Missing columns in game data: {missing_columns}")
            return None, None

        # Fill missing values
        games_data.fillna(0, inplace=True)
        player_stats.fillna(player_stats.mean(), inplace=True)

        logging.info("Preprocessing completed successfully.")
        return games_data, player_stats

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        st.error("Preprocessing failed. Check logs.")
        return None, None

# Train models
def train_models():
    games_data, player_stats = preprocess_data()
    if games_data is None or player_stats is None:
        st.error("Preprocessing failed. Training aborted.")
        return

    logging.info("Starting model training...")

    try:
        # Spread model
        X_games = games_data[['PTS', 'AST', 'REB']].values
        y_games = (games_data['WL'] == 'W').astype(int)  # Win = 1, Loss = 0

        X_train, X_test, y_train, y_test = train_test_split(
            X_games, y_games, test_size=0.2, random_state=42
        )

        spread_model = RandomForestRegressor(n_estimators=150, random_state=42)
        spread_model.fit(X_train, y_train)
        spread_rmse = mean_squared_error(y_test, spread_model.predict(X_test), squared=False)
        logging.info(f"Spread model RMSE: {spread_rmse:.2f}")

        # Player performance model
        X_players = player_stats[['PTS', 'AST', 'REB']].values
        y_players = player_stats['PTS']  # Predict points

        Xp_train, Xp_test, yp_train, yp_test = train_test_split(
            X_players, y_players, test_size=0.2, random_state=42
        )

        player_model = GradientBoostingRegressor(n_estimators=150)
        player_model.fit(Xp_train, yp_train)
        player_rmse = mean_squared_error(yp_test, player_model.predict(Xp_test), squared=False)
        logging.info(f"Player model RMSE: {player_rmse:.2f}")

        # Save models
        joblib.dump(spread_model, 'spread_model.pkl')
        joblib.dump(player_model, 'player_model.pkl')
        st.success("Models trained and saved successfully.")
        logging.info("Models saved to disk.")

    except Exception as e:
        logging.error(f"Model training failed: {e}")
        st.error("Model training failed. Check logs.")

# Predict outcomes
def predict_outcomes():
    logging.info("Predicting game outcomes...")

    try:
        games_data = pd.read_csv('nba_games.csv')
        spread_model = joblib.load('spread_model.pkl')
        player_model = joblib.load('player_model.pkl')
        predictions = []

        for _, game in games_data.iterrows():
            features = [[game['PTS'], game['AST'], game['REB']]]
            win_prob = spread_model.predict(features)[0]
            winner = "Home Team" if win_prob > 0.5 else "Visitor"

            player_stats = []
            player_stats_data = pd.read_csv('nba_player_stats.csv')
            for _, player in player_stats_data.iterrows():
                player_features = [[player['PTS'], player['AST'], player['REB']]]
                pred_pts = player_model.predict(player_features)[0]
                
                player_stats.append({
                    'Player': player['PLAYER_NAME'],
                    'Points': round(pred_pts, 1),
                    'Assists': round(pred_pts * 0.5, 1),
                    'Rebounds': round(pred_pts * 0.7, 1)
                })

            predictions.append({
                'Game': game['MATCHUP'],
                'Predicted Winner': winner,
                'Player Stats': player_stats
            })

        return predictions

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("Prediction failed. Check logs.")
        return []

# UI controls
if st.button("Fetch Data & Train Models"):
    with st.spinner("Fetching data and training models..."):
        fetch_data()
        train_models()

if st.button("Predict Game Outcomes"):
    with st.spinner("Predicting outcomes..."):
        predictions = predict_outcomes()
        for pred in predictions:
            st.subheader(f"Game: {pred['Game']}")
            st.write(f"**Predicted Winner:** {pred['Predicted Winner']}")
            st.table(pd.DataFrame(pred['Player Stats']))

st.info("Update data and models as needed.")
