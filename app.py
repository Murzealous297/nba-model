import streamlit as st
import pandas as pd
import joblib
import logging
from nba_api.stats.endpoints import leaguegamefinder, playergamelog, boxscoretraditionalv2
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

    # Fetch recent game data
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25')
    games = gamefinder.get_data_frames()[0]
    games.to_csv('nba_games.csv', index=False)

    # Example starting player IDs; modify as needed
    player_ids = ['2544', '201939']  # Replace with dynamic fetching if necessary
    all_stats = []
    
    for pid in player_ids:
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season='2024-25')
        all_stats.append(gamelog.get_data_frames()[0])
    
    player_data = pd.concat(all_stats)
    player_data.to_csv('nba_player_stats.csv', index=False)
    
    logging.info("Data fetching complete.")

# Preprocess data
def preprocess_data():
    games = pd.read_csv('nba_games.csv')
    players = pd.read_csv('nba_player_stats.csv')
    
    games['HOME_TEAM'] = games['HOME_TEAM'].astype('category').cat.codes
    players.fillna(players.mean(), inplace=True)
    
    return games, players

# Train models
def train_models():
    logging.info("Training models...")
    games, players = preprocess_data()
    
    # Spread Model
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
    logging.info(f"Player Metrics RMSE: {player_rmse}")

    joblib.dump(spread_model, 'spread_model.pkl')
    joblib.dump(player_model, 'player_model.pkl')
    
    logging.info("Models trained and saved.")

# Predict outcomes and player stats
def predict_outcomes():
    games = pd.read_csv('nba_games.csv')
    spread_model = joblib.load('spread_model.pkl')
    player_model = joblib.load('player_model.pkl')
    
    predictions = []
    for _, game in games.iterrows():
        game_features = game.drop(labels=['SPREAD']).values.reshape(1, -1)
        spread_prediction = spread_model.predict(game_features)[0]
        
        winner = game['HOME_TEAM_NAME'] if spread_prediction > 0 else game['VISITOR_TEAM_NAME']
        
        # Predict stats for starting players
        players = ['Player1_ID', 'Player2_ID']  # Replace with dynamic fetching if needed
        player_stats = []
        
        for pid in players:
            player_features = [0.5, 25, 5]  # Example features (modify as necessary)
            stats_pred = player_model.predict([player_features])[0]
            
            player_stats.append({
                'Player ID': pid,
                'Points': round(stats_pred, 1),
                'Assists': round(stats_pred * 0.5, 1),
                'Rebounds': round(stats_pred * 0.7, 1)
            })
        
        predictions.append({
            'Game': f"{game['HOME_TEAM_NAME']} vs {game['VISITOR_TEAM_NAME']}",
            'Predicted Winner': winner,
            'Player Stats': player_stats
        })
    
    return predictions

# Run button to fetch data and train models
if st.button("Run Data Fetch & Model Training"):
    with st.spinner("Fetching data and training models..."):
        fetch_data()
        train_models()
    st.success("Data fetched and models trained!")

# Display predictions
if st.button("Predict Outcomes & Player Stats"):
    with st.spinner("Making predictions..."):
        results = predict_outcomes()
        for result in results:
            st.subheader(f"Game: {result['Game']}")
            st.write(f"**Predicted Winner:** {result['Predicted Winner']}")
            
            stats_df = pd.DataFrame(result['Player Stats'])
            st.table(stats_df)

st.write("Data and models update manually. Predictions based on the latest models.")
