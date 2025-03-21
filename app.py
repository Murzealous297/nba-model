import streamlit as st
import pandas as pd
import joblib
import logging
from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Logging setup
logging.basicConfig(filename='nba_model.log', level=logging.INFO)

st.title("NBA Betting Prediction App")
st.write("Predict game outcomes and player stats (points, assists, rebounds) for starting players.")

# Function to fetch NBA data
def fetch_data():
    logging.info("Fetching NBA data...")

    # Fetch recent games
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25')
        games = gamefinder.get_data_frames()[0]
        games.to_csv('nba_games.csv', index=False)
        logging.info("Game data fetched and saved.")
    except Exception as e:
        logging.error(f"Error fetching game data: {e}")
        st.error("Error fetching game data. Check logs.")

    # Example player IDs; update as needed or fetch dynamically
    player_ids = ['2544', '201939']  # Example player IDs
    all_stats = []

    try:
        for pid in player_ids:
            gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024-25')
            all_stats.append(gamelog.get_data_frames()[0])

        player_data = pd.concat(all_stats)
        player_data.to_csv('nba_player_stats.csv', index=False)
        logging.info("Player stats fetched and saved.")
    except Exception as e:
        logging.error(f"Error fetching player stats: {e}")
        st.error("Error fetching player stats. Check logs.")

# Preprocess data with dynamic column checks
def preprocess_data():
    try:
        games = pd.read_csv('nba_games.csv')
        players = pd.read_csv('nba_player_stats.csv')

        # Dynamic check for home team column
        home_team_col = next((col for col in games.columns if 'HOME' in col and 'TEAM' in col), None)
        if not home_team_col:
            st.error("Home team column not found in game data.")
            logging.error("Home team column missing in games data.")
            return None, None
        games[home_team_col] = games[home_team_col].astype('category').cat.codes

        players.fillna(players.mean(), inplace=True)
        logging.info("Data preprocessing completed.")
        return games, players

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        st.error("Error preprocessing data. Check logs.")
        return None, None

# Train models for spread and player metrics
def train_models():
    games, players = preprocess_data()
    if games is None or players is None:
        st.error("Data preprocessing failed. Training aborted.")
        return

    logging.info("Starting model training...")

    try:
        # Train Spread Model
        X_games = games.drop(columns=['SPREAD'], errors='ignore')
        y_games = games['SPREAD']
        Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_games, y_games, test_size=0.2, random_state=42)

        spread_model = RandomForestRegressor(n_estimators=150, random_state=42)
        spread_model.fit(Xg_train, yg_train)
        
        spread_rmse = mean_squared_error(yg_test, spread_model.predict(Xg_test), squared=False)
        logging.info(f"Spread Model RMSE: {spread_rmse:.2f}")

        # Train Player Stats Model
        X_players = players.drop(columns=['PLAYER_METRIC'], errors='ignore')
        y_players = players['PLAYER_METRIC']
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_players, y_players, test_size=0.2, random_state=42)

        player_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)
        player_model.fit(Xp_train, yp_train)

        player_rmse = mean_squared_error(yp_test, player_model.predict(Xp_test), squared=False)
        logging.info(f"Player Metrics Model RMSE: {player_rmse:.2f}")

        # Save models
        joblib.dump(spread_model, 'spread_model.pkl')
        joblib.dump(player_model, 'player_model.pkl')
        st.success("Models trained and saved successfully!")
        logging.info("Models saved.")
    
    except Exception as e:
        logging.error(f"Model training error: {e}")
        st.error("Error during model training. Check logs.")

# Predict game outcomes and player stats
def predict_outcomes():
    try:
        games = pd.read_csv('nba_games.csv')
        spread_model = joblib.load('spread_model.pkl')
        player_model = joblib.load('player_model.pkl')
        predictions = []

        for _, game in games.iterrows():
            game_features = game.drop(labels=['SPREAD'], errors='ignore').values.reshape(1, -1)
            spread_pred = spread_model.predict(game_features)[0]
            
            winner = game['HOME_TEAM_NAME'] if spread_pred > 0 else game['VISITOR_TEAM_NAME']
            
            player_stats = []
            starting_players = ['2544', '201939']  # Example player IDs
            for pid in starting_players:
                # Example feature vector; customize as necessary
                player_features = [0.5, 25, 5]  # Example: [usage%, points, assists]
                stats_pred = player_model.predict([player_features])[0]
                
                player_stats.append({
                    'Player ID': pid,
                    'Predicted Points': round(stats_pred, 1),
                    'Predicted Assists': round(stats_pred * 0.5, 1),
                    'Predicted Rebounds': round(stats_pred * 0.7, 1)
                })

            predictions.append({
                'Game': f"{game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}",
                'Predicted Winner': winner,
                'Player Stats': player_stats
            })
        
        return predictions

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("Prediction error. Check logs.")
        return []

# UI Buttons and Functionality
if st.button("Run Data Fetch & Model Training"):
    with st.spinner("Fetching data and training models..."):
        fetch_data()
        train_models()

if st.button("Predict Game Outcomes & Player Stats"):
    with st.spinner("Making predictions..."):
        results = predict_outcomes()
        for result in results:
            st.subheader(f"Game: {result['Game']}")
            st.write(f"**Predicted Winner:** {result['Predicted Winner']}")
            
            stats_df = pd.DataFrame(result['Player Stats'])
            st.table(stats_df)

st.write("Models are updated manually. Predictions use the latest trained models.")
