from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

def scrape_player_data(profile_url, driver):
    """Scrape additional player data from their profile page."""
    driver.get(profile_url)
    time.sleep(5)  # Adjust if necessary

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Example of scraping additional data from the player's profile page
    player_data = {}
    print('Trying...')
    try:
        full_tier = soup.find('div', class_='leagueTier')
        lp = full_tier.find('span', class_='leaguePoints').get_text(strip=True)
        player_data["leagueTier"] = full_tier.get_text(strip=True).replace(lp, "")
        player_data["league_points"] = int(lp[:-3])
        player_data["wins"] = int(soup.find('div', class_='winslosses').find('span', class_='winsNumber').get_text(strip=True))
        player_data["loses"] = int(soup.find('div', class_='winslosses').find('span', class_='lossesNumber').get_text(strip=True))
        player_data["total_games"] =  player_data["wins"] + player_data["loses"]
        player_data["win_rate"] = round((100* (player_data["wins"] / player_data["total_games"])),1)

        # print("Waiting for table to load...")
        # table = soup.find("table", class_="data_table sortable_table")
        # print("Table found:", table is not None)
        #
        # if table:
        #     tbody = table.find("tbody")
        #     print("Tbody found:", tbody is not None)
        #
        #     if tbody:
        #         rows = tbody.find_all("tr")
        #         print(f"Number of rows found: {len(rows)}")
        #
        #         roles = []
        #         for row in rows:
        #             cols = row.find_all("td")
        #             print(f"Number of columns in this row: {len(cols)}")
        #
        #             if len(cols) >= 3:
        #                 role_data = {
        #                     "Role": cols[0].get_text(strip=True),
        #                     "Played": cols[1].get_text(strip=True),
        #                     "Winrate": cols[2].get_text(strip=True),
        #                 }
        #                 roles.append(role_data)
        #
        #         if roles:
        #             player_data["roles"] = roles
        #             for role in roles:
        #                 print(role)
        #         else:
        #             print("No roles data found.")
        #     else:
        #         print("No table body (tbody) found.")
        # else:
        #     print("Table not found.")

        # driver.get(profile_url)
        # wait = WebDriverWait(driver, 30)
        # print("Waiting for table to load...")
        # soup = BeautifulSoup(driver.page_source, "html.parser")
        # table = soup.find("table", class_="data_table sortable_table")
        # if table:
        #     tbody = table.find("tbody")
        #     if tbody:
        #         rows = tbody.find_all("tr")
        #         print("Rows found.")
        #
        #         roles = []
        #         for row in rows[1:6]:  # Skip header row
        #             cols = row.find_all("td")
        #             if len(cols) < 3:
        #                 continue  # Skip rows with missing columns
        #
        #             # Collect main player data
        #             role_data = {
        #                 "Role": cols[0].text.strip(),
        #                 "Played": cols[1].text.strip(),
        #                 "Winrate": cols[2].text.strip(),
        #             }
        #             roles.append(role_data)
        #
        #         if roles:
        #             for role in roles:
        #                 print(role)
        #         else:
        #             print("No roles data found.")
        #     else:
        #         print("No table body (tbody) found.")
        # else:
        #     print("Table not found.")


        # roles_table = WebDriverWait(driver, 10).until(
        #     EC.visibility_of_element_located(
        #         (By.CSS_SELECTOR, "#profileRoles .tabs-content .content.active .sortable_table tbody"))
        # )
        #
        # # Scrape the role, number of games played, and win rate for each role in solo queue
        # rows = roles_table.find_elements(By.TAG_NAME, "tr")
        # for row in rows:
        #     role = row.find_elements(By.TAG_NAME, "td")[0].text.strip()
        #     played = row.find_elements(By.TAG_NAME, "td")[1].get_attribute('data-sort-value').strip()
        #     winrate = row.find_elements(By.TAG_NAME, "td")[2].get_attribute('data-sort-value').strip()
        #
        #     print(f"Role: {role}, Played: {played}, Winrate: {winrate}")

        #TODO : get 2 best roles, and their winrate , soloq avg KDA , winrate with top 5 champs
        #TODO: in recent games?



    except AttributeError as e:
        print(f"Could not scrape data from {profile_url}. Error: {str(e)}")

    return player_data

def scrape():
    url = "https://www.leagueofgraphs.com/rankings/summoners"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    chrome_driver_path = "C:/Program Files/chromedriver-win64/chromedriver-win64/chromedriver.exe"
    service = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)

        # Wait for the table to be present
        wait = WebDriverWait(driver, 30)
        print("Waiting for table to load...")

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find the table
        table = soup.find("table", class_="data_table summonerRankingsTable with_sortable_column")
        if table:
            tbody = table.find("tbody")
            if tbody:
                rows = tbody.find_all("tr")
                print("Rows found.")

                players = []
                for row in rows[1:6]:  # Skip header row
                    cols = row.find_all("td")
                    if len(cols) < 4:
                        continue  # Skip rows with missing columns

                    player_url = cols[1].find("a")["href"]
                    player_full_url = f"https://www.leagueofgraphs.com{player_url}"

                    # Collect main player data
                    player_data = {
                        "Rank": cols[0].text.strip(),
                        "Summoner Name": cols[1].text.strip().split("\n")[0],
                        "Most Played Champions": [img["title"] for img in cols[4].find_all("img")],
                        "Profile URL": player_full_url
                    }

                    # Scrape additional player data
                    additional_data = scrape_player_data(player_full_url, driver)
                    player_data.update(additional_data)
                    print(player_data)
                    players.append(player_data)

                if players:
                    for player in players:
                        print(player)
                else:
                    print("No player data found.")
            else:
                print("No table body (tbody) found.")
        else:
            print("Table not found.")

    finally:
        driver.quit()

if __name__ == "__main__":
    scrape()
