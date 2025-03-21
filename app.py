import streamlit as st
import pandas as pd
import joblib
import logging
from basketball_reference_scraper import games, players
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Logging setup
logging.basicConfig(filename='nba_model.log', level=logging.INFO)

st.title("NBA Betting Prediction App (Basketball Reference)")
st.write("Predict NBA game outcomes and player stats (points, assists, rebounds).")

# Fetch NBA data from Basketball Reference
def fetch_data():
    logging.info("Fetching NBA data...")
    try:
        schedule = games.get_schedule('2025')
        schedule.to_csv('nba_schedule.csv', index=False)
        logging.info("Schedule data saved.")

        dates = schedule['DATE'].unique()[-5:]  # Last 5 game dates
        box_scores = []

        for date in dates:
            box = games.get_box_scores(date)
            box_scores.append(box)

        box_data = pd.concat(box_scores)
        box_data.to_csv('nba_box_scores.csv', index=False)
        logging.info("Box scores saved.")

        player_names = ['LeBron James', 'Stephen Curry']
        player_stats = []

        for player in player_names:
            stats = players.get_stats(player, stat_type='PER_GAME')
            player_stats.append(stats)

        stats_data = pd.concat(player_stats)
        stats_data.to_csv('nba_player_stats.csv', index=False)
        logging.info("Player stats saved.")
        st.success("Data fetched successfully!")

    except Exception as e:
        logging.error(f"Data fetch error: {e}")
        st.error("Error fetching data. Check logs.")

# Preprocess data
def preprocess_data():
    try:
        games_data = pd.read_csv('nba_schedule.csv')
        players_data = pd.read_csv('nba_player_stats.csv')

        st.write("### Game Data Columns")
        st.write(games_data.columns)

        home_col = next((col for col in games_data.columns if 'home' in col.lower()), None)
        visitor_col = next((col for col in games_data.columns if 'visitor' in col.lower()), None)

        if not home_col or not visitor_col:
            st.error("Missing team columns. Check logs.")
            logging.error(f"Missing columns. Available: {games_data.columns}")
            return None, None

        games_data[home_col] = games_data[home_col].astype('category').cat.codes
        players_data.fillna(players_data.mean(), inplace=True)
        logging.info("Data preprocessing completed.")
        return games_data, players_data

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        st.error("Preprocessing failed. Check logs.")
        return None, None

# Train models
def train_models():
    games_data, players_data = preprocess_data()
    if games_data is None or players_data is None:
        st.error("Preprocessing failed. Training aborted.")
        return

    try:
        X_games = games_data.drop(columns=['SPREAD'], errors='ignore')
        y_games = games_data['SPREAD']
        Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_games, y_games, test_size=0.2, random_state=42)

        spread_model = RandomForestRegressor(n_estimators=150, random_state=42)
        spread_model.fit(Xg_train, yg_train)
        spread_rmse = mean_squared_error(yg_test, spread_model.predict(Xg_test), squared=False)
        logging.info(f"Spread Model RMSE: {spread_rmse:.2f}")

        X_players = players_data.drop(columns=['PLAYER_METRIC'], errors='ignore')
        y_players = players_data['PLAYER_METRIC']
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_players, y_players, test_size=0.2, random_state=42)

        player_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)
        player_model.fit(Xp_train, yp_train)
        player_rmse = mean_squared_error(yp_test, player_model.predict(Xp_test), squared=False)
        logging.info(f"Player Model RMSE: {player_rmse:.2f}")

        joblib.dump(spread_model, 'spread_model.pkl')
        joblib.dump(player_model, 'player_model.pkl')
        st.success("Models trained and saved!")
        logging.info("Models saved.")

    except Exception as e:
        logging.error(f"Training error: {e}")
        st.error("Training failed. Check logs.")

# Predict outcomes
def predict_outcomes():
    try:
        games_data = pd.read_csv('nba_schedule.csv')
        spread_model = joblib.load('spread_model.pkl')
        player_model = joblib.load('player_model.pkl')
        predictions = []

        for _, game in games_data.iterrows():
            game_features = game.drop(labels=['SPREAD'], errors='ignore').values.reshape(1, -1)
            spread_pred = spread_model.predict(game_features)[0]
            
            winner = game['HOME_TEAM'] if spread_pred > 0 else game['VISITOR_TEAM']
            player_stats = []

            for player in ['LeBron James', 'Stephen Curry']:
                stats_pred = player_model.predict([[0.5, 25, 5]])[0]  # Example features
                player_stats.append({
                    'Player': player,
                    'Points': round(stats_pred, 1),
                    'Assists': round(stats_pred * 0.5, 1),
                    'Rebounds': round(stats_pred * 0.7, 1)
                })

            predictions.append({
                'Game': f"{game['HOME_TEAM']} vs {game['VISITOR_TEAM']}",
                'Winner': winner,
                'Player Stats': player_stats
            })

        return predictions

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("Prediction failed. Check logs.")
        return []

# UI Controls
if st.button("Fetch Data & Train Models"):
    with st.spinner("Fetching data and training models..."):
        fetch_data()
        train_models()

if st.button("Predict Game Outcomes"):
    with st.spinner("Making predictions..."):
        results = predict_outcomes()
        for res in results:
            st.subheader(f"Game: {res['Game']}")
            st.write(f"**Predicted Winner:** {res['Winner']}")
            st.table(pd.DataFrame(res['Player Stats']))

st.info("Models are updated manually. Predictions use the latest models.")
