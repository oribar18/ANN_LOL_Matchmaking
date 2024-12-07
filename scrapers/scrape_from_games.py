import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


# Initialize the Selenium WebDriver
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


# Function to scrape player-specific page data
def scrape_player_page(driver, url):
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.content[data-tab-id="championsData-soloqueue"]'))
        )
        # KDA under Soloqueue
        try:
            kda_section = driver.find_element(By.CSS_SELECTOR,
                                              'div.content.active[data-tab-id="championsData-soloqueue"] div.number')
            kills = kda_section.find_element(By.CLASS_NAME, 'kills').text
            deaths = kda_section.find_element(By.CLASS_NAME, 'deaths').text
            assists = kda_section.find_element(By.CLASS_NAME, 'assists').text
            kda = f"{kills}/{deaths}/{assists}"
        except Exception:
            kda = "N/A"

        # Most Played Role
        try:
            most_played_role = driver.find_element(By.CSS_SELECTOR,
                                                   'div.content.active[data-tab-id="championsData-soloqueue"] td a div.txt.name').text.strip()
        except Exception:
            most_played_role = "N/A"

        # Winrate and Wins
        try:
            winrate_elements = driver.find_elements(By.CSS_SELECTOR, 'div.pie-chart.small')
            winrate = "N/A"
            for element in winrate_elements:
                if '%' in element.text:
                    winrate_text = element.text.strip()
                    winrate_percentage = winrate_text.strip('%')
                    break
            wins_element = driver.find_element(By.CSS_SELECTOR, 'span.wins span.winsNumber')
            wins = wins_element.text.strip()
            winrate = f"Wins: {wins} ({winrate_percentage}%)" if wins and winrate_percentage else "N/A"
        except Exception:
            winrate = "N/A"

        # LP (League Points)
        try:
            lp_element = driver.find_element(By.CSS_SELECTOR, 'span.leaguePoints')
            lp_text = lp_element.text.strip().replace('LP', '').strip()
            lp = int(lp_text) if lp_text else "N/A"
        except Exception:
            lp = "N/A"

        return {
            'kda': kda,
            'most_played_role': most_played_role,
            'winrate': winrate,
            'lp': lp
        }

    except Exception as e:
        return {
            'kda': "N/A",
            'most_played_role': "N/A",
            'winrate': "N/A",
            'lp': "N/A"
        }


# Function to scrape champion-specific stats
def scrape_champion_stats(driver, url, num_of_champs=5):
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.data_table tbody tr'))
        )
        rows = driver.find_elements(By.CSS_SELECTOR, 'table.data_table tbody tr')[1:]
        total_creeps, total_gold, count = 0, 0, 0
        for row in rows:
            try:
                columns = row.find_elements(By.TAG_NAME, 'td')
                if len(columns) >= 6:
                    creeps = float(columns[4].text.strip())
                    gold = float(columns[5].text.strip())
                    total_creeps += creeps
                    total_gold += gold
                    count += 1
                    if count == num_of_champs:
                        break
            except Exception:
                continue
        avg_creeps = total_creeps / count if count > 0 else "N/A"
        avg_gold = total_gold / count if count > 0 else "N/A"
        return {'avg_creeps_per_min': avg_creeps, 'avg_gold_per_min': avg_gold}
    except TimeoutException:
        print(f"Timeout while loading champion stats: {url}")
        return {'avg_creeps_per_min': "N/A", 'avg_gold_per_min': "N/A"}


# Main scraping function
def scrape_players_from_games(file_path):
    games_df = pd.read_csv(file_path)
    player_columns = [col for col in games_df.columns if 'relative href' in col]
    players_data = []
    driver = get_driver()

    for _, row in games_df.iterrows():
        for i, player_col in enumerate(player_columns, start=1):  # Enumerate for dynamic column handling
            relative_href = row[player_col]
            if pd.notna(relative_href):  # Skip if the value is NaN
                try:
                    # Scrape player data
                    player_data = scrape_player_page(driver, relative_href)

                    # Scrape champion stats
                    champion_stats = scrape_champion_stats(
                        driver, relative_href.replace('/summoner/', '/summoner/champions/')
                    )
                    player_data.update(champion_stats)

                    # Add username and combined rank_lp dynamically
                    name_col = f'name {i}' if i > 1 else 'name'
                    subname_col = f'subname {i}' if i > 1 else 'subname'
                    player_data['username'] = row[name_col]
                    player_data['rank_lp'] = f"{row[subname_col]} {player_data['lp']} LP"

                    players_data.append(player_data)

                except Exception as e:
                    print(f"Error processing {relative_href}: {e}")

    driver.quit()

    # Save the collected data to a CSV
    players_df = pd.DataFrame(players_data)
    players_df = players_df.drop(columns=['lp'], axis=1)
    players_df.to_csv('games_players_data.csv', index=False)
    print(players_df)


# Run the script
if __name__ == '__main__':
    scrape_players_from_games('../data/games_data_raw.csv')
