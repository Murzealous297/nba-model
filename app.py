import streamlit as st
import pandas as pd
import joblib
import logging
from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from apscheduler.schedulers.background import BackgroundScheduler

# Logging setup
logging.basicConfig(filename='nba_model.log', level=logging.INFO)

# Streamlit setup
st.title("NBA Betting Model")
st.write("Predict NBA game spreads and player performance.")

# Background scheduler for automatic data fetching
scheduler = BackgroundScheduler()
scheduler.start()

def fetch_data():
    logging.info("Fetching NBA data...")
    
    # Fetch game data
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25')
    games = gamefinder.get_data_frames()[0]
    games.to_csv('nba_games.csv', index=False)
    
    # Example player IDs (replace with dynamic fetching if needed)
    player_ids = ['2544', '201939']
    all_stats = []
    for pid in player_ids:
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024-25')
        all_stats.append(gamelog.get_data_frames()[0])
    
    player_data = pd.concat(all_stats)
    player_data.to_csv('nba_player_stats.csv', index=False)
    
    logging.info("Data fetched and saved.")

def preprocess_data():
    games = pd.read_csv('nba_games.csv')
    players = pd.read_csv('nba_player_stats.csv')
    
    games['HOME_TEAM'] = games['HOME_TEAM'].astype('category').cat.codes
    players.fillna(players.mean(), inplace=True)
    
    return games, players

def train_models():
    logging.info("Training models...")
    
    games, players = preprocess_data()
    
    # Game Spread Model
    X_games = games.drop(columns=['SPREAD'])
    y_games = games['SPREAD']
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_games, y_games, test_size=0.2)
    
    spread_model = RandomForestRegressor(n_estimators=150)
    spread_model.fit(Xg_train, yg_train)
    
    spread_rmse = mean_squared_error(yg_test, spread_model.predict(Xg_test), squared=False)
    logging.info(f"Spread RMSE: {spread_rmse}")
    
    # Player Metrics Model
    X_players = players.drop(columns=['PLAYER_METRIC'])
    y_players = players['PLAYER_METRIC']
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_players, y_players, test_size=0.2)
    
    player_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)
    player_model.fit(Xp_train, yp_train)
    
    player_rmse = mean_squared_error(yp_test, player_model.predict(Xp_test), squared=False)
    logging.info(f"Player RMSE: {player_rmse}")
    
    joblib.dump(spread_model, 'spread_model.pkl')
    joblib.dump(player_model, 'player_model.pkl')
    logging.info("Models saved.")

@scheduler.scheduled_job('interval', hours=24)
def scheduled_update():
    fetch_data()
    train_models()

# Schedule data fetch and model training
if st.button("Update Data & Retrain Models"):
    scheduled_update()
    st.success("Data updated and models retrained.")

# Prediction UI
st.subheader("Predict Game Spread")
spread_input = st.text_input("Enter features for spread prediction (comma-separated):")
if st.button("Predict Spread"):
    try:
        features = list(map(float, spread_input.split(',')))
        spread_model = joblib.load("spread_model.pkl")
        prediction = spread_model.predict([features])[0]
        st.success(f"Predicted Spread: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.subheader("Predict Player Performance")
player_input = st.text_input("Enter features for player prediction (comma-separated):")
if st.button("Predict Player Metric"):
    try:
        features = list(map(float, player_input.split(',')))
        player_model = joblib.load("player_model.pkl")
        prediction = player_model.predict([features])[0]
        st.success(f"Predicted Player Metric: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.write("App is running. Models are updated every 24 hours.")
