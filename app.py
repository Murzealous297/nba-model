import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

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

# Function to fetch advanced metrics from FiveThirtyEight
def fetch_advanced_metrics():
    url = "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
    response = requests.get(url)
    elo_data = pd.read_csv(StringIO(response.text))
    return elo_data

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
    elo_data = fetch_advanced_metrics()
    todays_games = fetch_todays_games()
    return team_stats, player_stats, elo_data, todays_games

# Streamlit App
st.title("NBA Betting Data Automation")

# Run button
if st.button("Run"):
    team_stats, player_stats, elo_data, todays_games = fetch_all_data()
    
    # Display fetched data
    st.write("Team Stats:")
    st.write(team_stats.head())
    
    st.write("Player Stats:")
    st.write(player_stats.head())
    
    st.write("Advanced Metrics (ELO):")
    st.write(elo_data.head())
    
    st.write("Today's Games:")
    st.write(todays_games)
