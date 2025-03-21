import streamlit as st
import pandas as pd
import joblib
import logging
from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logging.basicConfig(filename='nba_model.log', level=logging.INFO)

st.title("NBA Betting Prediction App (nba_api)")
st.write("Predict NBA game outcomes and player stats (points, assists, rebounds).")

# Fetch NBA data using nba_api
def fetch_data():
    logging.info("Fetching NBA data with nba_api...")
    try:
        # Fetch last season's game data
        gamefinder = leaguegamefinder.LeagueGameFinder(season_type_nullable='Regular Season')
        games_data = gamefinder.get_data_frames()[0]
        games_data.to_csv('nba_games.csv', index=False)
        logging.info("Games data fetched and saved.")

        # Fetch recent player stats
        players = ['LeBron James', 'Stephen Curry']
        all_player_stats = []

        for player in players:
            gamelog = playergamelog.PlayerGameLog(player_id=player, season='2024-25', season_type_all_star='Regular Season')
            stats = gamelog.get_data_frames()[0]
            all_player_stats.append(stats)

        player_stats = pd.concat(all_player_stats)
        player_stats.to_csv('nba_player_stats.csv', index=False)
        logging.info("Player stats fetched and saved.")
        
        st.success("Data fetched successfully!")

    except Exception as e:
        logging.error(f"Data fetch error: {e}")
        st.error("Data fetching failed. Check logs.")

# Preprocess data
def preprocess_data():
    try:
        games_data = pd.read_csv('nba_games.csv')
        player_stats = pd.read_csv('nba_player_stats.csv')

        st.write("### Game Data Columns")
        st.write(games_data.columns)

        required_columns = ['GAME_ID', 'MATCHUP', 'WL', 'PTS', 'AST', 'REB']
        if not all(col in games_data.columns for col in required_columns):
            st.error("Required columns missing in game data.")
            logging.error(f"Columns available: {games_data.columns}")
            return None, None

        games_data.fillna(0, inplace=True)
        player_stats.fillna(player_stats.mean(), inplace=True)
        logging.info("Preprocessing complete.")
        return games_data, player_stats

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        st.error("Preprocessing failed. Check logs.")
        return None, None

# Train models
def train_models():
    games_data, player_stats = preprocess_data()
    if games_data is None or player_stats is None:
        st.error("Preprocessing failed. Aborting training.")
        return

    try:
        # Spread Model
        X_games = games_data[['PTS', 'AST', 'REB']].values
        y_games = (games_data['WL'] == 'W').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X_games, y_games, test_size=0.2, random_state=42)

        spread_model = RandomForestRegressor(n_estimators=150, random_state=42)
        spread_model.fit(X_train, y_train)
        spread_rmse = mean_squared_error(y_test, spread_model.predict(X_test), squared=False)
        logging.info(f"Spread Model RMSE: {spread_rmse:.2f}")

        # Player Performance Model
        X_players = player_stats[['PTS', 'AST', 'REB']].values
        y_players = player_stats['PTS']  # Predict points
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_players, y_players, test_size=0.2, random_state=42)

        player_model = GradientBoostingRegressor(n_estimators=150)
        player_model.fit(Xp_train, yp_train)
        player_rmse = mean_squared_error(yp_test, player_model.predict(Xp_test), squared=False)
        logging.info(f"Player Model RMSE: {player_rmse:.2f}")

        joblib.dump(spread_model, 'spread_model.pkl')
        joblib.dump(player_model, 'player_model.pkl')
        st.success("Models trained and saved!")

    except Exception as e:
        logging.error(f"Training error: {e}")
        st.error("Training failed. Check logs.")

# Predict outcomes
def predict_outcomes():
    try:
        games_data = pd.read_csv('nba_games.csv')
        spread_model = joblib.load('spread_model.pkl')
        player_model = joblib.load('player_model.pkl')
        predictions = []

        for _, game in games_data.iterrows():
            game_features = [[game['PTS'], game['AST'], game['REB']]]
            win_prob = spread_model.predict(game_features)[0]
            predicted_winner = "Home Team" if win_prob > 0.5 else "Visitor"

            player_predictions = []
            for _, player in pd.read_csv('nba_player_stats.csv').iterrows():
                stats_features = [[player['PTS'], player['AST'], player['REB']]]
                predicted_points = player_model.predict(stats_features)[0]

                player_predictions.append({
                    'Player': player['PLAYER_NAME'],
                    'Points': round(predicted_points, 1),
                    'Assists': round(predicted_points * 0.5, 1),
                    'Rebounds': round(predicted_points * 0.7, 1)
                })

            predictions.append({
                'Game': game['MATCHUP'],
                'Predicted Winner': predicted_winner,
                'Player Stats': player_predictions
            })

        return predictions

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("Prediction failed. Check logs.")
        return []

# UI Controls
if st.button("Fetch Data & Train Models"):
    with st.spinner("Fetching data and training..."):
        fetch_data()
        train_models()

if st.button("Predict Game Outcomes"):
    with st.spinner("Predicting..."):
        predictions = predict_outcomes()
        for pred in predictions:
            st.subheader(f"Game: {pred['Game']}")
            st.write(f"**Winner Prediction:** {pred['Predicted Winner']}")
            st.table(pd.DataFrame(pred['Player Stats']))

st.info("Data and models are updated on each run.")
