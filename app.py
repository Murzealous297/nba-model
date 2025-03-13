import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Function to fetch team stats
def fetch_team_stats(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Scrape team stats
    table = soup.find('table', {'id': 'per_game-team'})
    headers = [th.text for th in table.find('thead').find_all('th')]
    data = [[td.text for td in row.find_all('td')] for row in table.find('tbody').find_all('tr')]
    team_stats = pd.DataFrame(data, columns=headers[1:])
    return team_stats

# Function to fetch player stats
def fetch_player_stats(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Scrape player stats
    table = soup.find('table', {'id': 'per_game_stats'})
    headers = [th.text for th in table.find('thead').find_all('th')]
    data = [[td.text for td in row.find_all('td')] for row in table.find('tbody').find_all('tr')]
    player_stats = pd.DataFrame(data, columns=headers[1:])
    return player_stats

# Function to fetch today's games
def fetch_todays_games():
    url = "https://www.basketball-reference.com/boxscores/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    games = []
    for game in soup.find_all('div', class_='game_summary'):
        teams = game.find_all('a')
        if len(teams) >= 2:
            away_team = teams[0].text
            home_team = teams[1].text
            games.append({'Away Team': away_team, 'Home Team': home_team})
    return pd.DataFrame(games)

# Cache data to avoid re-fetching
@st.cache_data
def fetch_all_data():
    st.write("Fetching all data...")
    team_stats = fetch_team_stats(2023)  # Replace with current year
    player_stats = fetch_player_stats(2023)
    todays_games = fetch_todays_games()
    return team_stats, player_stats, todays_games

# Function to recommend player points over/under bets
def recommend_player_bets(player_stats):
    # Clean and preprocess player stats
    player_stats['PTS'] = pd.to_numeric(player_stats['PTS'], errors='coerce')
    player_stats = player_stats.dropna(subset=['PTS'])
    
    # Sort players by points per game (descending)
    top_players = player_stats.sort_values(by='PTS', ascending=False).head(10)
    
    # Recommend top 3 players for over/under bets
    recommendations = []
    for _, row in top_players.iterrows():
        player_name = row['Player']
        pts = row['PTS']
        recommendations.append(f"{player_name} (Avg PTS: {pts:.1f}) - Over/Under: {pts + 2:.1f}")
    
    return recommendations[:3]  # Return top 3 recommendations

# Function to predict win/loss for today's games
def predict_win_loss(todays_games, team_stats):
    # Clean and preprocess team stats
    team_stats['PTS'] = pd.to_numeric(team_stats['PTS'], errors='coerce')
    team_stats = team_stats.dropna(subset=['PTS'])
    
    # Create a dictionary of team points
    team_points = dict(zip(team_stats['Team'], team_stats['PTS']))
    
    # Predict win/loss for each game
    predictions = []
    for _, game in todays_games.iterrows():
        away_team = game['Away Team']
        home_team = game['Home Team']
        
        # Get team points (default to 0 if team not found)
        away_pts = team_points.get(away_team, 0)
        home_pts = team_points.get(home_team, 0)
        
        # Predict winner
        if away_pts > home_pts:
            predictions.append(f"{away_team} (Away) vs {home_team} (Home) - Predicted Winner: {away_team}")
        else:
            predictions.append(f"{away_team} (Away) vs {home_team} (Home) - Predicted Winner: {home_team}")
    
    return predictions

# Streamlit App
st.title("JT NBA Betting Model")

# Run button
if st.button("Run"):
    # Initialize progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Update progress
    def update_progress(percentage):
        progress_bar.progress(percentage)
        progress_text.text(f"Progress: {percentage}%")

    # Fetch all data with progress updates
    update_progress(0)  # Start at 0%
    
    # Fetch team stats
    team_stats = fetch_team_stats(2023)
    update_progress(33)  # 33% complete after fetching team stats
    
    # Fetch player stats
    player_stats = fetch_player_stats(2023)
    update_progress(66)  # 66% complete after fetching player stats
    
    # Fetch today's games
    todays_games = fetch_todays_games()
    update_progress(100)  # 100% complete after fetching today's games
    
    # Display fetched data
    st.write("Team Stats:")
    st.write(team_stats.head())
    
    st.write("Player Stats:")
    st.write(player_stats.head())
    
    st.write("Today's Games:")
    st.write(todays_games)
    
    # Summary Section
    st.write("---")
    st.header("Bet Recommendations")
    
    # Bet 1: Player Points Over/Under Bets
    st.subheader("Bet 1: Top 3 Player Points Over/Under Bets")
    player_bets = recommend_player_bets(player_stats)
    for bet in player_bets:
        st.write(f"- {bet}")
    
    # Bet 2: Win/Loss Predictions
    st.subheader("Bet 2: Win/Loss Predictions for Today's Games")
    win_loss_predictions = predict_win_loss(todays_games, team_stats)
    for prediction in win_loss_predictions:
        st.write(f"- {prediction}")
