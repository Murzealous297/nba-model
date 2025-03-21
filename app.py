import streamlit as st
import pandas as pd
import joblib
import logging
from basketball_reference_scraper.games import get_schedule, get_box_scores
from basketball_reference_scraper.players import get_stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Logging setup
logging.basicConfig(filename='nba_model.log', level=logging.INFO)

st.title("NBA Betting Prediction App (Basketball Reference)")
st.write("Predict NBA game outcomes and player stats (points, assists, rebounds).")

# Function to fetch NBA data from Basketball Reference
def fetch_data():
    logging.info("Fetching NBA data from Basketball Reference...")

    try:
        # Fetch schedule for the 2024-25 season
        schedule = get_schedule('2025')
        schedule.to_csv('nba_schedule.csv', index=False)
        logging.info("Schedule data fetched and saved.")

        # Fetch box scores for recent games
        dates = schedule['DATE'].unique()[-5:]  # Last 5 unique game dates for simplicity
        box_scores = []

        for date in dates:
            box = get_box_scores(date)
            box_scores.append(box)

        box_data = pd.concat(box_scores)
        box_data.to_csv('nba_box_scores.csv', index=False)
        logging.info("Box score data fetched and saved.")

        # Example player stats fetching (modify player names dynamically if needed)
        players = ['LeBron James', 'Stephen Curry']
        all_stats = []

        for player in players:
            stats = get_stats(player, stat_type='PER_GAME')
            all_stats.append(stats)

        player_stats = pd.concat(all_stats)
        player_stats.to_csv('nba_player_stats.csv', index=False)
        logging.info("Player stats fetched and saved.")
        
        st.success("Data fetched successfully!")

    except Exception as e:
        logging.error(f"Data fetch error: {e}")
        st.error("Error fetching data. Check logs.")

# Preprocess data with dynamic column detection
def preprocess_data():
    try:
        games = pd.read_csv('nba_schedule.csv')
        players = pd.read_csv('nba_player_stats.csv')

        st.write("### Game Data Columns")
        st.write(games.columns)  # Debug: Display columns

        home_team_col = next((col for col in games.columns if 'home' in col.lower()), None)
        visitor_team_col = next((col for col in games.columns if 'visitor' in col.lower()), None)

        if not home_team_col or not visitor_team_col:
            st.error("Required columns not found in game data.")
            logging.error(f"Missing columns. Available columns: {games.columns}")
            return None, None

        games[home_team_col] = games[home_team_col].astype('category').cat.codes
        players.fillna(players.mean(), inplace=True)
        
        logging.info(f"Preprocessing complete with columns: {home_team_col}, {visitor_team_col}.")
        return games, players

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        st.error("Error during data preprocessing. Check logs.")
        return None, None

# Train models
def train_models():
    games, players = preprocess_data()
    if games is None or players is None:
        st.error("Preprocessing failed. Model training aborted.")
        return

    try:
        # Spread Model
        X_games = games.drop(columns=['SPREAD'], errors='ignore')
        y_games = games['SPREAD']
        Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_games, y_games, test_size=0.2, random_state=42)

        spread_model = RandomForestRegressor(n_estimators=150, random_state=42)
        spread_model.fit(Xg_train, yg_train)
        
        spread_rmse = mean_squared_error(yg_test, spread_model.predict(Xg_test), squared=False)
        logging.info(f"Spread Model RMSE: {spread_rmse:.2f}")

        # Player Metrics Model
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
        
        st.success("Models trained and saved!")
        logging.info("Models saved successfully.")

    except Exception as e:
        logging.error(f"Model training error: {e}")
        st.error("Error during model training. Check logs.")

# Predict outcomes
def predict_outcomes():
    try:
        games = pd.read_csv('nba_schedule.csv')
        spread_model = joblib.load('spread_model.pkl')
        player_model = joblib.load('player_model.pkl')
        predictions = []

        for _, game in games.iterrows():
            game_features = game.drop(labels=['SPREAD'], errors='ignore').values.reshape(1, -1)
            spread_pred = spread_model.predict(game_features)[0]
            
            winner = game['HOME_TEAM'] if spread_pred > 0 else game['VISITOR_TEAM']
            
            player_stats = []
            players = ['LeBron James', 'Stephen Curry']  # Example players
            for player in players:
                features = [0.5, 25, 5]  # Example features: usage%, points, assists
                stats_pred = player_model.predict([features])[0]

                player_stats.append({
                    'Player': player,
                    'Predicted Points': round(stats_pred, 1),
                    'Predicted Assists': round(stats_pred * 0.5, 1),
                    'Predicted Rebounds': round(stats_pred * 0.7, 1)
                })
            
            predictions.append({
                'Game': f"{game['HOME_TEAM']} vs {game['VISITOR_TEAM']}",
                'Predicted Winner': winner,
                'Player Stats': player_stats
            })
        
        return predictions

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("Prediction error. Check logs.")
        return []

# UI Controls
if st.button("Fetch Data & Train Models"):
    with st.spinner("Fetching data and training models..."):
        fetch_data()
        train_models()

if st.button("Predict Outcomes"):
    with st.spinner("Predicting game outcomes..."):
        results = predict_outcomes()
        for result in results:
            st.subheader(f"Game: {result['Game']}")
            st.write(f"**Predicted Winner:** {result['Predicted Winner']}")
            st.table(pd.DataFrame(result['Player Stats']))

st.write("Models are manually updated. Predictions use the latest models.")
