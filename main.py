from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException
import pandas as pd
import time


# Set up Selenium WebDriver
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")  # For resource issues

    chrome_options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


# Function to scrape champion-specific stats for a player
def scrape_champion_stats(driver, username, num_of_champs):
    username = username.replace('#', '-')
    url = f'https://www.leagueofgraphs.com/summoner/champions/sg/{username}'
    driver.get(url)

    try:
        # Wait for the champion stats table to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.data_table tbody tr'))
        )

        # Locate the rows of the champion table and skip the first row (header)
        rows = driver.find_elements(By.CSS_SELECTOR, 'table.data_table tbody tr')[1:]  # Skip the header row

        total_creeps = 0
        total_gold = 0
        count = 0

        for row in rows:
            try:
                columns = row.find_elements(By.TAG_NAME, 'td')
                if len(columns) >= 6:  # Ensure enough columns exist
                    # Extract Creeps / minute (5th column) and Gold / min (6th column)
                    creeps_text = columns[4].text.strip()
                    gold_text = columns[5].text.strip()

                    # Ensure values are valid floats
                    creeps = float(creeps_text) if creeps_text else 0
                    gold = float(gold_text) if gold_text else 0

                    total_creeps += creeps
                    total_gold += gold
                    count += 1

                    if count == num_of_champs:  # Stop if we reach the desired number of champions
                        break
            except Exception as e:
                print(f"Error extracting champion stats for {username} in row {count + 1}: {e}")
                continue

        # Calculate averages
        avg_creeps = total_creeps / count if count > 0 else "N/A"
        avg_gold = total_gold / count if count > 0 else "N/A"

        return {'avg_creeps_per_min': avg_creeps, 'avg_gold_per_min': avg_gold}

    except TimeoutException:
        print(f"Timeout while waiting for champion stats table for {username}.")
        return {'avg_creeps_per_min': "N/A", 'avg_gold_per_min': "N/A"}
    except Exception as e:
        print(f"Error scraping champion stats for {username}: {e}")
        return {'avg_creeps_per_min': "N/A", 'avg_gold_per_min': "N/A"}


# Function to scrape player-specific page data
def scrape_player_page(driver, username):
    username = username.replace('#', '-')
    url = f'https://www.leagueofgraphs.com/summoner/sg/{username}'
    driver.get(url)

    try:
        # Wait for the Soloqueue section to load
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
        except Exception as e:
            print(f"Error retrieving KDA for {username}: {e}")
            kda = "N/A"

        # Most Played Role
        try:
            most_played_role = driver.find_element(By.CSS_SELECTOR,
                                                   'div.content.active[data-tab-id="championsData-soloqueue"] td a div.txt.name').text.strip()
        except Exception as e:
            print(f"Error retrieving most played role for {username}: {e}")
            most_played_role = "N/A"

        return {
            'kda': kda,
            'most_played_role': most_played_role
        }

    except Exception as e:
        print(f"Error scraping {username}'s page: {e}")
        return {}


# Scrape player profiles data from multiple pages
def scrape_player_profiles():
    base_url = 'https://www.leagueofgraphs.com/rankings/summoners/sg/page-'
    driver = get_driver()

    # List to store player data
    players_data = []

    # Loop through pages
    for page_num in range(1, 51):  # Adjust range as needed
        url = f'{base_url}{page_num}'
        driver.get(url)

        # Explicit wait to ensure the table loads properly
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "data_table"))
            )
        except TimeoutException:
            print(f"Timeout while waiting for page {page_num} to load.")
            continue

        # Extract row data (username, rank, winrate, etc.)
        player_rows = driver.find_elements(By.CSS_SELECTOR, 'table.data_table tbody tr')
        row_data = []
        for row in player_rows:
            try:
                username = row.find_element(By.CSS_SELECTOR, 'span.name').text.strip()
                rank_lp = row.find_element(By.CSS_SELECTOR, 'td div.summonerTier').text.strip()
                winrate = row.find_element(By.CSS_SELECTOR, 'td div.wins').text.strip()
                champions = [champ.get_attribute('alt') for champ in row.find_elements(By.CSS_SELECTOR, 'td img')]
                row_data.append({
                    'username': username,
                    'rank_lp': rank_lp,
                    'winrate': winrate,
                    'most_played_champions': champions
                })
            except NoSuchElementException:
                print(f"Row skipped due to missing elements.")
                continue

        # Navigate to each player's page and collect additional data
        for player in row_data:
            username = player['username']
            if username == 'N/A':
                continue
            player_page_data = scrape_player_page(driver, username)
            player.update(player_page_data)
            players_data.append(player)

            # Scrape champion stats for averages
            champion_stats = scrape_champion_stats(driver, username, len(champions))
            player.update(champion_stats)

            # Return to the main page and reload
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "data_table"))
            )

        print(f"Finished scraping page {page_num}")

    driver.quit()
    return players_data


# Main function
def main():
    players = scrape_player_profiles()

    # Create DataFrame and save results
    df = pd.DataFrame(players)
    df.to_csv('league_of_graphs_players.csv', index=False)
    print(df)


if __name__ == '__main__':
    main()
